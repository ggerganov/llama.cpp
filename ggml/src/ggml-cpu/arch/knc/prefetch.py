#!/usr/bin/env python3
"""Generate candidate-only prefetch schedules from the pinned KNC sources."""
import re


def profile_guard(text):
    return text.replace(
        '#if defined(TARGET_KNC_3120A) == defined(TARGET_KNC_7120P)',
        '#if (defined(TARGET_KNC_3120A) + defined(TARGET_KNC_5110P) + defined(TARGET_KNC_7120P)) != 1')


def q5_schedule(text, l1, l2, step, q8):
    text = profile_guard(text)
    text, replacements = re.subn(r'#ifdef TARGET_KNC_3120A\n#define KNC_L1_ITERS 4\n#define KNC_L2_ITERS 16\n#endif\n'
                  r'#ifdef TARGET_KNC_7120P\n#define KNC_L1_ITERS 8\n#define KNC_L2_ITERS 32\n#endif',
                  f'#define KNC_L1_ITERS {l1}\n#define KNC_L2_ITERS {l2}', text)
    if replacements != 1:
        raise ValueError('Pinned Q5 assembly profile definitions changed')
    for name, distance in [('KNC_L1_ITERS', l1), ('KNC_L2_ITERS', l2)]:
        if not distance:
            text = re.sub(r'    KNC_PREFETCH_PACKED \\at, ' + name + r', vprefetch[01]\n', '', text)
    if step == 64:
        for offset in (16, 32, 48):
            text = text.replace('    KNC_PREFETCH_STREAM (\\q8off+%d)\n' % offset, '')
    if not q8:
        text = text.replace('    \\hint (((\\at+16*\\distance)/256)*292+4+(\\at+16*\\distance)%256)(%r9)\n', '')
    return text


def extended_schedule(text, l1, l2, step, q8):
    """Keep arithmetic instructions unchanged; replace only generated prefetch groups."""
    text = profile_guard(text)
    start = text.index('#ifdef TARGET_KNC_3120A')
    end = text.index('.text', start)
    text = text[:start] + text[end:]
    pattern = re.compile(r'#ifdef TARGET_KNC_3120A\n.*?#endif\n#ifdef TARGET_KNC_7120P\n.*?#endif', re.S)
    seen = set()

    def replace(match):
        label_match = re.search(r'\.Lskip([2346])_(\d+)_', match[0])
        if label_match is None:
            raise ValueError('Pinned assembly prefetch label changed')
        q, group = map(int, label_match.groups())
        if (q, group) in seen:
            raise ValueError('Duplicate prefetch group')
        seen.add((q, group))
        p = group * 16
        if p % step:
            return ''
        lines = []
        for level, distance in enumerate((l1, l2)):
            if not distance:
                continue
            pos = p + distance * 16
            b, i = divmod(pos, 256)
            stride = {2: 84, 3: 110, 4: 144, 6: 210}[q]
            if q in (2, 3):
                offsets = [b*stride + (16 if q == 2 else 32) + i//128*32 + i%32]
                if q == 3:
                    offsets.append(b*stride + i%32)
            elif q == 4:
                offsets = [b*stride + 16 + i//64*32 + i%32]
            else:
                offsets = [b*stride + i//128*64 + i%128//32%2*32 + i%32,
                           b*stride + 128 + i//128*32 + i%32]
            label = f'.Lmq_skip{q}_{group}_{level}'
            lines += [f'cmpq ${b}, %r10', f'jbe {label}']
            lines += [f'vprefetch{level} {offset}(%r8)' for offset in offsets]
            if q8:
                lines += [f'vprefetch{level} {b*292+4+i}(%r9)']
            lines += [label + ':']
        return '\n'.join(lines)

    text = pattern.sub(replace, text)
    if len(seen) != 64:
        raise ValueError('Pinned assembly prefetch groups changed')
    return text


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--l1', type=int, required=True)
    p.add_argument('--l2', type=int, required=True)
    p.add_argument('--step', type=int, choices=(16,64), required=True)
    p.add_argument('--q8', choices=('ON','OFF'), required=True)
    a = p.parse_args()
    if not 0 <= a.l1 <= 256 or not 0 <= a.l2 <= 256:
        p.error('Prefetch lead must be in 0..256')
    for name, generate in [('ggml-knc-native.S',q5_schedule),('quant-native.S',extended_schedule)]:
        output = a.output/name
        text = generate((a.source/name).read_text(encoding='utf-8'),a.l1,a.l2,a.step,a.q8=='ON')
        if not output.exists() or output.read_text(encoding='utf-8') != text:
            with output.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
