#!/usr/bin/env python3

"""
This library provides a translation from the Wasm binary format to Kast.
"""

import sys
import json
import kwasm_ast as a

from wasm.parsers.module import parse_module, Module
from wasm.datatypes import ValType, TypeIdx, FunctionType, Function, Table, TableType, Memory, MemoryType, Limits, Global, GlobalType, Mutability, ElementSegment, DataSegment, StartFunction, Import, Export
from wasm.opcodes import BinaryOpcode

def main():
    if len(list(sys.argv)) == 1:
        infile = sys.stdin
    else:
        infile = open(sys.argv[1], 'rb')
    module = wasm2kast(infile)
    infile.close()
    return module

def wasm2kast(wasm_bytes : bytes, filename=None) -> dict:
    """Returns a dictionary representing the Kast JSON."""
    ast = parse_module(wasm_bytes)
    return ast2kast(ast, filename=filename)

def ast2kast(wasm_ast : Module, filename=None) -> dict:
    """Returns a dictionary representing the Kast JSON."""
    types   = a.defns([typ(x)    for x in wasm_ast.types])
    funcs   = a.defns([func(x)   for x in wasm_ast.funcs])
    tables  = a.defns([table(x)  for x in wasm_ast.tables])
    mems    = a.defns([memory(x) for x in wasm_ast.mems])
    globs   = a.defns([glob(x)   for x in wasm_ast.globals])
    elems   = a.defns([elem(x)   for x in wasm_ast.elem])
    datas   = a.defns([data(x)   for x in wasm_ast.data])
    starts  = a.defns(start(wasm_ast.start))
    imports = a.defns([imp(x)    for x in wasm_ast.imports])
    exports = a.defns([export(x) for x in wasm_ast.exports])
    meta    = a.module_metadata(filename=filename)
    return a.module(types=types, funcs=funcs, tables=tables, mems=mems, globs=globs, elem=elems, data=datas, start=starts, imports=imports, exports=exports, metadata=meta)

#########
# Defns #
#########

def typ(t : FunctionType):
    return a.type(func_type(t.params, t.results))

def func(f : Function):
    type = a.KInt(f.type_idx)
    ls_list = [val_type(x) for x in f.locals]
    locals = a.vec_type(a.val_types(ls_list))
    body = instrs(f.body)
    return a.func(type, locals, body)

def table(t : Table):
    ls = limits(t.type.limits)
    return a.table(ls)

def memory(m : Memory):
    ls = limits(m.type)
    return a.memory(ls)

def glob(g : Global):
    t = global_type(g.type)
    init = instrs(g.init)
    return a.glob(t, init)

def elem(e : ElementSegment):
    offset = instrs(e.offset)
    return a.elem(e.table_idx, offset, e.init)

def data(d : DataSegment):
    offset = instrs(d.offset)
    return a.data(d.memory_idx, offset, d.init)

def start(s : StartFunction):
    if s is None:
        return []
    return [a.start(s.function_idx)]

def imp(i : Import):
    mod_name = a.wasm_string(i.module_name)
    name = a.wasm_string(i.as_name)
    t = type(i.desc)
    if t is TypeIdx:
        desc = a.func_desc(i.desc)
    elif t is GlobalType:
        desc = a.global_desc(global_type(i.desc))
    elif t is TableType:
        desc = a.table_desc(limits(i.desc.limits))
    elif t is MemoryType:
        desc = a.memory_desc(limits(i.desc))
    return a.imp(mod_name, name, desc)

def export(e : Export):
    name = a.wasm_string(e.name)
    idx = e.desc
    return a.export(name, idx)

##########
# Instrs #
##########

block_id = 0

def instrs(iis):
    """Turn a list of instructions into KAST."""
    # We ignore `END`.
    # The AST supplied by py-wasm has already parsed these and terminated the blocks.
    # We also ignore `ELSE`.
    # The AST supplied by py-wasm includes the statements in the else-branch as part of the `IF` instruction.
    return a.instrs([instr(i) for i in iis
                                            if not i.opcode == BinaryOpcode.END
                                           and not i.opcode == BinaryOpcode.ELSE])

def instr(i):
    B = BinaryOpcode
    global block_id
    if i.opcode == B.BLOCK:
        cur_block_id = block_id
        block_id += 1
        iis = instrs(i.instructions)
        res = vec_type(i.result_type)
        return a.BLOCK(res, iis, a.KInt(cur_block_id))
    if i.opcode == B.BR:
        return a.BR(i.label_idx)
    if i.opcode == B.BR_IF:
        return a.BR_IF(i.label_idx)
    if i.opcode == B.BR_TABLE:
        return a.BR_TABLE(i.label_indices, i.default_idx)
    if i.opcode == B.CALL:
        return a.CALL(i.function_idx)
    if i.opcode == B.CALL_INDIRECT:
        return a.CALL_INDIRECT(i.type_idx)
    if i.opcode == B.ELSE:
        raise(ValueError("ELSE opcode: should have been filtered out."))
    if i.opcode == B.END:
        raise(ValueError("End opcode: should have been filtered out."))
    if i.opcode == B.F32_CONST:
        return a.F32_CONST(i.value)
    if i.opcode == B.F64_CONST:
        return a.F64_CONST(i.value)
    if i.opcode == B.F32_REINTERPRET_I32:
        raise(ValueError('Reinterpret instructions not implemented.'))
    if i.opcode == B.F64_REINTERPRET_I64:
        raise(ValueError('Reinterpret instructions not implemented.'))
    if i.opcode == B.GET_GLOBAL:
        return a.GET_GLOBAL(i.global_idx)
    if i.opcode == B.GET_LOCAL:
        return a.GET_LOCAL(i.local_idx)
    if i.opcode == B.I32_CONST:
        return a.I32_CONST(i.value)
    if i.opcode == B.I64_CONST:
        return a.I64_CONST(i.value)
    if i.opcode == B.I32_REINTERPRET_F32:
        raise(ValueError('Reinterpret instructions not implemented.'))
    if i.opcode == B.I64_REINTERPRET_F64:
        raise(ValueError('Reinterpret instructions not implemented.'))
    if i.opcode == B.IF:
        cur_block_id = block_id
        block_id += 1
        thens = instrs(i.instructions)
        els = instrs(i.else_instructions)
        res = vec_type(i.result_type)
        return a.IF(res, thens, els, a.KInt(cur_block_id))
    if i.opcode == B.F32_STORE:
        return a.F32_STORE(i.memarg.offset)
    if i.opcode == B.F64_STORE:
        return a.F64_STORE(i.memarg.offset)
    if i.opcode == B.I32_STORE:
        return a.I32_STORE(i.memarg.offset)
    if i.opcode == B.I64_STORE:
        return a.I64_STORE(i.memarg.offset)
    if i.opcode == B.I32_STORE16:
        return a.I32_STORE16(i.memarg.offset)
    if i.opcode == B.I64_STORE16:
        return a.I64_STORE16(i.memarg.offset)
    if i.opcode == B.I32_STORE8:
        return a.I32_STORE8(i.memarg.offset)
    if i.opcode == B.I64_STORE8:
        return a.I64_STORE8(i.memarg.offset)
    if i.opcode == B.I64_STORE32:
        return a.I64_STORE32(i.memarg.offset)
    if i.opcode == B.F32_LOAD:
        return a.F32_LOAD(i.memarg.offset)
    if i.opcode == B.F64_LOAD:
        return a.F64_LOAD(i.memarg.offset)
    if i.opcode == B.I32_LOAD:
        return a.I32_LOAD(i.memarg.offset)
    if i.opcode == B.I64_LOAD:
        return a.I64_LOAD(i.memarg.offset)
    if i.opcode == B.I32_LOAD16_S:
        return a.I32_LOAD16_S(i.memarg.offset)
    if i.opcode == B.I32_LOAD16_U:
        return a.I32_LOAD16_U(i.memarg.offset)
    if i.opcode == B.I64_LOAD16_S:
        return a.I64_LOAD16_S(i.memarg.offset)
    if i.opcode == B.I64_LOAD16_U:
        return a.I64_LOAD16_U(i.memarg.offset)
    if i.opcode == B.I32_LOAD8_S:
        return a.I32_LOAD8_S(i.memarg.offset)
    if i.opcode == B.I32_LOAD8_U:
        return a.I32_LOAD8_U(i.memarg.offset)
    if i.opcode == B.I64_LOAD8_S:
        return a.I64_LOAD8_S(i.memarg.offset)
    if i.opcode == B.I64_LOAD8_U:
        return a.I64_LOAD8_U(i.memarg.offset)
    if i.opcode == B.I64_LOAD32_S:
        return a.I64_LOAD32_S(i.memarg.offset)
    if i.opcode == B.I64_LOAD32_U:
        return a.I64_LOAD32_U(i.memarg.offset)
    if i.opcode == B.LOOP:
        cur_block_id = block_id
        block_id += 1
        iis = instrs(i.instructions)
        res = vec_type(i.result_type)
        return a.LOOP(res, iis, a.KInt(cur_block_id))
    if i.opcode == B.SET_GLOBAL:
        return a.SET_GLOBAL(i.global_idx)
    if i.opcode == B.SET_LOCAL:
        return a.SET_LOCAL(i.local_idx)
    if i.opcode == B.TEE_LOCAL:
        return a.TEE_LOCAL(i.local_idx)
    # Catch all for operations without direct arguments.
    op = i.opcode
    return eval('a.' + i.opcode.name)

########
# Data #
########

def val_type(t : ValType):
    if t == ValType.i32:
        return a.i32
    if t == ValType.i64:
        return a.i64
    if t == ValType.f32:
        return a.f32
    if t == ValType.f64:
        return a.f64

def vec_type(ts : [ValType]):
    _ts = [val_type(x) for x in ts]
    return a.vec_type(a.val_types(_ts))

def func_type(params, results):
    pvec = vec_type(params)
    rvec = vec_type(results)
    return a.func_type(pvec, rvec)

def limits(l : Limits):
    return (l.min, l.max)

def global_type(t : GlobalType):
    vt = val_type(t.valtype)
    if t.mut is Mutability.const:
        return a.global_type(a.MUT_CONST, vt)
    return a.global_type(a.MUT_VAR, vt)

########
# Main #
########

if __name__ == "__main__":
    main()
