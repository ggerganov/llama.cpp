#!/usr/bin/env python3
"""Translate the narrow production .S instruction subset to host test semantics.

This is deliberately not a KNC compiler, ISA emulator, or inference fallback.
It executes the source's actual addresses, registers, branches and arithmetic
through instrumented scalar C, against the independent dequantization oracle.
Native assembly/linking and execution remain separate validation gates.
"""
import argparse
import ast
import hashlib
import pathlib
import re
import subprocess


def constant(expression):
    """Evaluate GAS integer expressions without accepting arbitrary Python."""
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)): return left // right
            if isinstance(node.op, ast.Mod): return left % right
            if isinstance(node.op, ast.LShift): return left << right
            if isinstance(node.op, ast.RShift): return left >> right
            if isinstance(node.op, ast.BitAnd): return left & right
            if isinstance(node.op, ast.BitOr): return left | right
        raise ValueError('unsupported integer expression: ' + expression)
    return evaluate(ast.parse(expression, mode='eval').body)


def arguments(text):
    """Split operands without splitting indexed memory or decorations."""
    depth = 0
    result, current = [], []
    for char in text:
        if char in '({': depth += 1
        if char in ')}': depth -= 1
        if char == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        result.append(''.join(current).strip())
    return result


REGISTERS = 'rax rcx rdx rbx rsp rbp rsi rdi r8 r9 r10 r11 r12 r13 r14 r15'.split()


def scalar_register(operand):
    name = operand[1:] if operand.startswith('%') else operand
    if name == 'eax': name = 'rax'
    if name not in REGISTERS: raise ValueError('unreviewed scalar register ' + operand)
    return f'g[{REGISTERS.index(name)}]'


def vector_register(operand):
    match = re.fullmatch(r'%zmm([0-9]+)', operand)
    if match is None or int(match[1]) >= 32:
        raise ValueError('unreviewed vector operand ' + operand)
    return f'z[{int(match[1])}]'


def memory(operand):
    match = re.fullmatch(r'(.*)\((%[a-z0-9]+(?:,%[a-z0-9]+,[1248])?)\)(?:\{([a-z0-9]+)\})?', operand)
    if match is None: raise ValueError('unreviewed memory operand ' + operand)
    displacement, address, conversion = match.groups()
    if address == '%rip':
        if re.fullmatch(r'\.L[A-Za-z0-9_]+', displacement) is None:
            raise ValueError('unreviewed RIP symbol ' + displacement)
        return f'(uintptr_t)&{displacement[1:]}', conversion
    address_parts = address.split(',')
    result = scalar_register(address_parts[0])
    if len(address_parts) == 3:
        result += f' + {scalar_register(address_parts[1])} * {address_parts[2]}u'
    if displacement:
        result += f' + (uintptr_t)({constant(displacement)})'
    return f'({result})', conversion


def scalar(operand):
    if operand.startswith('$'): return f'(uintptr_t)({constant(operand[1:])})'
    if operand.startswith('%'): return scalar_register(operand)
    address, conversion = memory(operand)
    if conversion is not None: raise ValueError('scalar converted load')
    return f'model_load_u64({address})'


def instruction(line):
    head = line.split(None, 1)
    name = head[0]
    operands = arguments(head[1]) if len(head) > 1 else []
    if name == 'vpsubd':
        a,b,d=map(vector_register,operands)
        return f'for (lane=0; lane<16; ++lane) {d}.u[lane] = {b}.u[lane] - {a}.u[lane];'
    if name == 'ret': return 'return;'
    if name in ('movq', 'addq', 'subq'):
        if len(operands) != 2: raise ValueError(line)
        op = {'movq': '=', 'addq': '+=', 'subq': '-='}[name]
        return f'{scalar_register(operands[1])} {op} {scalar(operands[0])};'
    if name == 'xorl':
        if operands != ['%eax', '%eax']: raise ValueError('only reviewed 32-bit zero idiom: ' + line)
        return 'g[0] = 0;'
    if name == 'cmpq':
        return f'left = {scalar(operands[1])}; right = {scalar(operands[0])};'
    if name in ('jb', 'jbe', 'jae'):
        if len(operands) != 1 or not re.fullmatch(r'\.L[A-Za-z0-9_]+', operands[0]):
            raise ValueError('unreviewed branch: ' + line)
        comparison = {'jb': '<', 'jbe': '<=', 'jae': '>='}[name]
        return f'if (left {comparison} right) goto {operands[0][1:]};'
    if name in ('vprefetch0', 'vprefetch1'):
        address, conversion = memory(operands[0])
        if conversion is not None: raise ValueError('converted prefetch')
        hint = '_MM_HINT_T0' if name == 'vprefetch0' else '_MM_HINT_T1'
        return f'ggml_knc_test_prefetch((const void *){address}, {hint});'
    if name in ('vpxord', 'vpandd', 'vpord', 'vaddps', 'vmulps', 'vfmadd231ps'):
        if len(operands) != 3: raise ValueError(line)
        a, b, d = map(vector_register, operands)
        if name in ('vpxord', 'vpandd', 'vpord'):
            operator = {'vpxord': '^', 'vpandd': '&', 'vpord': '|'}[name]
            expression = f'{b}.u[lane] {operator} {a}.u[lane]'
            field = 'u'
        else:
            expression = (f'fmaf({b}.f[lane], {a}.f[lane], {d}.f[lane])'
                          if name == 'vfmadd231ps' else
                          f'{b}.f[lane] ' + ('+' if name == 'vaddps' else '*') + f' {a}.f[lane]')
            field = 'f'
        return f'for (lane = 0; lane < 16; ++lane) {d}.{field}[lane] = {expression};'
    if name in ('vpsrld', 'vpslld'):
        if len(operands) != 3 or not operands[0].startswith('$'): raise ValueError(line)
        count = constant(operands[0][1:])
        if count < 0 or count > 31: raise ValueError('unreviewed shift count: ' + line)
        operator = '>>' if name == 'vpsrld' else '<<'
        source, dest = map(vector_register, operands[1:])
        return f'for (lane = 0; lane < 16; ++lane) {dest}.u[lane] = {source}.u[lane] {operator} {count};'
    if name in ('vmovaps', 'vmovdqa32'):
        if len(operands) != 2: raise ValueError(line)
        if operands[0].startswith('%'):
            source = vector_register(operands[0])
            address, conversion = memory(operands[1])
            if conversion is not None: raise ValueError('unreviewed converted store')
            return f'model_store({address}, &{source});'
        address, conversion = memory(operands[0])
        dest = vector_register(operands[1])
        if conversion is None:
            return f'model_load(&{dest}, {address});'
        if name == 'vmovdqa32' and conversion == 'uint8':
            return f'model_load_u8(&{dest}, {address});'
        if name == 'vmovaps' and conversion == 'sint8':
            return f'model_load_i8_f32(&{dest}, {address});'
        raise ValueError('unreviewed converted vector load: ' + line)
    if name == 'vpbroadcastd':
        address, conversion = memory(operands[0])
        if conversion is not None: raise ValueError('unreviewed broadcast conversion')
        return f'model_broadcast(&{vector_register(operands[1])}, {address});'
    if name in ('vcvtfxpntdq2ps', 'vcvtfxpntps2dq'):
        if len(operands) != 3 or operands[0] != '$0':
            raise ValueError('only no-exponent-adjust conversion is reviewed: ' + line)
        dest = vector_register(operands[2])
        if name == 'vcvtfxpntdq2ps' and not operands[1].startswith('%'):
            address, conversion = memory(operands[1])
            if conversion != '1to16': raise ValueError('unreviewed conversion memory form')
            return f'model_broadcast_i32_f32(&{dest}, {address});'
        source = vector_register(operands[1])
        helper = 'model_i32_f32' if name == 'vcvtfxpntdq2ps' else 'model_f32_i32'
        return f'{helper}(&{dest}, &{source});'
    raise ValueError('unreviewed instruction: ' + line)


def translate(source):
    lines = expand_macros(source)
    functions, data = {}, {}
    current = None
    data_label = None
    for line in lines:
        if line.startswith('.section'):
            current = None
            continue
        if line.startswith('.long '):
            if data_label is None: raise ValueError('unlabelled data')
            data[data_label] = constant(line.split(None, 1)[1])
            data_label = None
            continue
        if line.endswith(':'):
            label = line[:-1]
            if label.startswith('ggml_knc_'):
                current = []
                functions[label] = current
            elif current is not None:
                current.append(line)
            else:
                data_label = label
            continue
        if line.startswith(('.text', '.p2align', '.globl', '.type', '.size')):
            continue
        if line.startswith('.'):
            raise ValueError('unreviewed assembly directive: ' + line)
        if current is None: raise ValueError('instruction outside function: ' + line)
        current.append(line)
    if set(functions) not in ({f'ggml_knc_q{q}_block_native' for q in (2,3,4,6)},
                              {'ggml_knc_q5_block_native', 'ggml_knc_f32_vectors_native'}):
        raise ValueError('unexpected assembly entry points')
    output = ['/* GENERATED HOST TEST MODEL; NEVER A NATIVE KNC BINARY. */',
              '#include "imci-test-shim.h"']
    for label, value in data.items():
        if re.fullmatch(r'\.L[A-Za-z0-9_]+', label) is None: raise ValueError(label)
        output.append(f'static const uint32_t {label[1:]} = {value};')
    for name, body in functions.items():
        if name.endswith('_block_native'):
            output += [
                f'void {name}(const uint8_t *q4, const uint8_t *qh,',
                '    const int8_t *q8, const int32_t *scales, const uint8_t *x,',
                '    const uint8_t *y, size_t remaining, int32_t *lanes) {',
                '    uintptr_t model_stack[3] = {0, remaining, (uintptr_t)lanes};',
                '    uintptr_t g[16] = {0};',
                '    g[7]=(uintptr_t)q4; g[6]=(uintptr_t)qh; g[2]=(uintptr_t)q8;',
                '    g[1]=(uintptr_t)scales; g[8]=(uintptr_t)x; g[9]=(uintptr_t)y;',
                '    g[4]=(uintptr_t)model_stack;']
        else:
            output += [
                'void ggml_knc_f32_vectors_native(size_t n, const float *x,',
                '    const float *y, float *lanes) {',
                '    uintptr_t g[16] = {0};',
                '    g[7]=n; g[6]=(uintptr_t)x; g[2]=(uintptr_t)y; g[1]=(uintptr_t)lanes;']
        output += ['    model_vector z[32]; unsigned lane;',
                   '    uintptr_t left = 0, right = 0;',
                   '    memset(z, 0xa5, sizeof(z));']
        for line in body:
            if line.endswith(':'):
                output.append(line[1:] + ';')
            else:
                output.append('    /* ' + line + ' */')
                output.append('    ' + instruction(line))
        output.append('}')
    return '\n'.join(output) + '\n', len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default='gcc')
    parser.add_argument('--profile', required=True, choices=('3120A', '5110P', '7120P'))
    parser.add_argument('source', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    # These identities are ONLY for preprocessing .S text for this host model.
    # They are never passed to a C compilation or to the native build script.
    source = subprocess.check_output([
        args.cc, '-E', '-P', '-x', 'assembler-with-cpp', '-D__k1om__',
        '-D__KNC__', '-DTARGET_KNC_' + args.profile, str(args.source)], text=True)
    generated, count = translate(source)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.write_text('/* Production .S SHA256: ' + digest + ' */\n' + generated)
    print(f'HOST SOURCE SEMANTICS: {args.profile}, {count} expanded lines, source SHA256 {digest}')


def expand_macros(source):
    """Expand only .macro/.endm parameters and GNU assembler's unique \\@."""
    macros, lines = {}, []
    pending = None
    for line in source.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith('.macro '):
            if pending is not None: raise ValueError('nested macro definition')
            head = line.split(None, 2)
            pending = (head[1], arguments(head[2]) if len(head) > 2 else [], [])
        elif line == '.endm':
            if pending is None: raise ValueError('unmatched .endm')
            macros[pending[0]] = pending[1:]
            pending = None
        elif pending is not None:
            pending[2].append(line)
        else:
            lines.append(line)
    if pending is not None: raise ValueError('unterminated .macro')
    invocation = 0

    def expand(line, depth=0):
        nonlocal invocation
        if depth > 16: raise ValueError('recursive assembly macros')
        head = line.split(None, 1)
        if head[0] not in macros:
            yield line
            return
        names, body = macros[head[0]]
        values = arguments(head[1]) if len(head) > 1 else []
        if len(names) != len(values): raise ValueError('macro arity: ' + line)
        invocation += 1
        unique = invocation
        mapping = dict(zip(names, values))
        for inner in body:
            inner = inner.replace('\\@', str(unique))
            inner = re.sub(r'\\([A-Za-z_][A-Za-z_0-9]*)',
                           lambda match: mapping[match[1]], inner)
            yield from expand(inner, depth + 1)
    result = [expanded for line in lines for expanded in expand(line)]
    if any('\\' in line for line in result):
        raise ValueError('unexpanded assembly parameter')
    return result


if __name__ == '__main__':
    main()
