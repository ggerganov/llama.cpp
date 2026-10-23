# Convert RWKV7 (g1k) .pth checkpoints to llama.cpp GGUF with BF16X weights.
#
# Two-phase, resumable (long single passes get killed on some setups):
#   python convert_rwkv7_to_gguf.py <pth> <vocab> <out> meta
#   python convert_rwkv7_to_gguf.py <pth> <vocab> <out> data <first> <last_excl>
#
# Matrix tensors (ndim >= 2, ne0 % 32 == 0, >= 64K elements) are written as
# GGML_TYPE_BF16X (11.5 bpw); all 1-D parameters (norms, lerp vectors,
# w0/a0/v0, k_k/k_a/r_k) are kept in f32 for numerical safety.
from __future__ import annotations
import io
import struct
import sys
import numpy as np
import torch

sys.path.insert(0, r"F:\bfloat16x")
from bf16x_gguf_block import quantize_tensor

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_ALIGN = 32

T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY = 4, 5, 6, 7, 8, 9
GGML_F32, GGML_BF16X = 0, 43

def w_str(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

def kv_str(f, k, v):
    w_str(f, k); f.write(struct.pack('<I', T_STRING)); w_str(f, v)

def kv_u32(f, k, v):
    w_str(f, k); f.write(struct.pack('<I', T_U32)); f.write(struct.pack('<I', v))

def kv_f32(f, k, v):
    w_str(f, k); f.write(struct.pack('<I', T_F32)); f.write(struct.pack('<f', v))

def kv_bool(f, k, v):
    w_str(f, k); f.write(struct.pack('<I', T_BOOL)); f.write(struct.pack('<B', 1 if v else 0))

def kv_arr_f32(f, k, arr):
    w_str(f, k); f.write(struct.pack('<I', T_ARRAY)); f.write(struct.pack('<I', T_F32))
    f.write(struct.pack('<Q', len(arr)))
    f.write(np.asarray(arr, dtype='<f4').tobytes())

def kv_arr_i32(f, k, arr):
    w_str(f, k); f.write(struct.pack('<I', T_ARRAY)); f.write(struct.pack('<I', T_I32))
    f.write(struct.pack('<Q', len(arr)))
    f.write(np.asarray(arr, dtype='<i4').tobytes())

def kv_arr_str(f, k, arr):
    w_str(f, k); f.write(struct.pack('<I', T_ARRAY)); f.write(struct.pack('<I', T_STRING))
    f.write(struct.pack('<Q', len(arr)))
    for s in arr:
        w_str(f, s)

def build_plan(sd):
    keys = list(sd.keys())
    n_layer = 1 + max(int(k.split('.')[1]) for k in keys if k.startswith('blocks.'))
    n_embd = int(sd['emb.weight'].shape[1])
    n_vocab = int(sd['emb.weight'].shape[0])
    n_ff = int(sd['blocks.0.ffn.key.weight'].shape[0])
    rank_decay = int(sd['blocks.0.att.w1'].shape[1])
    rank_iclr = int(sd['blocks.0.att.a1'].shape[1])
    rank_v = int(sd['blocks.1.att.v1'].shape[1])
    rank_gate = int(sd['blocks.0.att.g1'].shape[1])

    plan = []
    def add(name, ne, gtype, pth_key=None, arr=None, zeros=False):
        plan.append((name, ne, gtype, pth_key, arr, zeros))

    add('token_embd.weight', [n_embd, n_vocab], GGML_BF16X, pth_key='emb.weight')
    add('token_embd_norm.weight', [n_embd], GGML_F32, pth_key='blocks.0.ln0.weight')
    add('token_embd_norm.bias', [n_embd], GGML_F32, pth_key='blocks.0.ln0.bias')
    add('output_norm.weight', [n_embd], GGML_F32, pth_key='ln_out.weight')
    add('output_norm.bias', [n_embd], GGML_F32, pth_key='ln_out.bias')
    add('output.weight', [n_embd, n_vocab], GGML_BF16X, pth_key='head.weight')

    for i in range(n_layer):
        p = f'blocks.{i}.'
        g = f'blk.{i}.'
        add(g + 'attn_norm.weight', [n_embd], GGML_F32, pth_key=p + 'ln1.weight')
        add(g + 'attn_norm.bias', [n_embd], GGML_F32, pth_key=p + 'ln1.bias')
        add(g + 'attn_norm_2.weight', [n_embd], GGML_F32, pth_key=p + 'ln2.weight')
        add(g + 'attn_norm_2.bias', [n_embd], GGML_F32, pth_key=p + 'ln2.bias')

        # 6-way fused token-shift lerp: r, w, k, v, a, g
        lerp_key = (p, [f'att.x_{c}' for c in 'rwkvag'], 'lerp_fused')
        add(g + 'time_mix_lerp_fused.weight', [n_embd, 1, 1, 6], GGML_F32,
            pth_key=lerp_key)

        add(g + 'time_mix_w0.weight', [n_embd], GGML_F32, pth_key=p + 'att.w0')
        add(g + 'time_mix_w1.weight', [n_embd, rank_decay], GGML_BF16X, pth_key=p + 'att.w1')
        add(g + 'time_mix_w2.weight', [rank_decay, n_embd], GGML_BF16X, pth_key=p + 'att.w2')
        add(g + 'time_mix_a0.weight', [n_embd], GGML_F32, pth_key=p + 'att.a0')
        add(g + 'time_mix_a1.weight', [n_embd, rank_iclr], GGML_BF16X, pth_key=p + 'att.a1')
        add(g + 'time_mix_a2.weight', [rank_iclr, n_embd], GGML_BF16X, pth_key=p + 'att.a2')
        add(g + 'time_mix_v0.weight', [n_embd], GGML_F32, pth_key=p + 'att.v0')
        if i == 0:
            # layer 0's value-residual params are unused by the graph; the
            # loader expects them at the iclr rank, zero-filled
            add(g + 'time_mix_v1.weight', [n_embd, rank_iclr], GGML_BF16X, zeros=True)
            add(g + 'time_mix_v2.weight', [rank_iclr, n_embd], GGML_BF16X, zeros=True)
        else:
            add(g + 'time_mix_v1.weight', [n_embd, rank_v], GGML_BF16X, pth_key=p + 'att.v1')
            add(g + 'time_mix_v2.weight', [rank_v, n_embd], GGML_BF16X, pth_key=p + 'att.v2')
        add(g + 'time_mix_g1.weight', [n_embd, rank_gate], GGML_BF16X, pth_key=p + 'att.g1')
        add(g + 'time_mix_g2.weight', [rank_gate, n_embd], GGML_BF16X, pth_key=p + 'att.g2')
        add(g + 'time_mix_k_k.weight', [n_embd], GGML_F32, pth_key=p + 'att.k_k')
        add(g + 'time_mix_k_a.weight', [n_embd], GGML_F32, pth_key=p + 'att.k_a')
        add(g + 'time_mix_r_k.weight', [n_embd], GGML_F32, pth_key=p + 'att.r_k')
        add(g + 'time_mix_key.weight', [n_embd, n_embd], GGML_BF16X, pth_key=p + 'att.key.weight')
        add(g + 'time_mix_value.weight', [n_embd, n_embd], GGML_BF16X, pth_key=p + 'att.value.weight')
        add(g + 'time_mix_receptance.weight', [n_embd, n_embd], GGML_BF16X, pth_key=p + 'att.receptance.weight')
        add(g + 'time_mix_ln.weight', [n_embd], GGML_F32, pth_key=p + 'att.ln_x.weight')
        add(g + 'time_mix_ln.bias', [n_embd], GGML_F32, pth_key=p + 'att.ln_x.bias')
        add(g + 'time_mix_output.weight', [n_embd, n_embd], GGML_BF16X, pth_key=p + 'att.output.weight')
        add(g + 'channel_mix_lerp_k.weight', [n_embd], GGML_F32, pth_key=p + 'ffn.x_k')
        add(g + 'channel_mix_key.weight', [n_embd, n_ff], GGML_BF16X, pth_key=p + 'ffn.key.weight')
        add(g + 'channel_mix_value.weight', [n_ff, n_embd], GGML_BF16X, pth_key=p + 'ffn.value.weight')

    hparams = dict(n_layer=n_layer, n_embd=n_embd, n_vocab=n_vocab, n_ff=n_ff,
                   rank_decay=rank_decay, rank_iclr=rank_iclr, rank_v=rank_v,
                   rank_gate=rank_gate)
    return plan, hparams

def lerp_fused_np(sd, key):
    p, names, _ = key
    # ggml tensor shape is (n_embd, 1, 1, 6) with ne0-fastest data layout, so the
    # six mixer vectors must be stored as CONTIGUOUS PLANES:
    #   data[i + n_embd * j] = x_<name_j>[i]
    # (an axis=-1 stack would interleave them per element and corrupt every
    # mixer vector)
    vecs = [sd[p + n].detach().to(torch.float32).numpy() for n in names]
    return np.stack(vecs, axis=0)  # (6, n_embd) -> flat = plane-major

def main():
    argv = sys.argv[1:]
    pth, vocab_file, out_path = argv[0], argv[1], argv[2]
    mode = argv[3] if len(argv) > 3 else 'meta'
    first = int(argv[4]) if len(argv) > 4 else 0
    last = int(argv[5]) if len(argv) > 5 else 10 ** 9

    print(f'loading {pth}...', flush=True)
    if pth.endswith('.safetensors'):
        from safetensors.torch import load_file
        raw = load_file(pth)
        sd = {(k[6:] if k.startswith('rwkv7.') else k): v for k, v in raw.items()}
    else:
        sd = torch.load(pth, map_location='cpu', mmap=True, weights_only=True)
    plan, hp = build_plan(sd)
    n_tensors = len(plan)
    print(f"n_layer={hp['n_layer']} n_embd={hp['n_embd']} n_vocab={hp['n_vocab']} "
          f"n_ff={hp['n_ff']} tensors={n_tensors}", flush=True)

    # ---- vocab: txt ids are 1-based; GGUF ids shift down by 1 ----
    # The World vocab file contains ~10.9k duplicate byte sequences (the same
    # bytes stored both as '...' escapes and as b'...' byte tokens). llama.cpp
    # requires a 1:1 token-string mapping, so the first occurrence keeps the
    # id's real token and every later duplicate gets a unique UNUSED marker.
    def unescape(s):
        # matches the reference tokenizer: Python-string semantics (eval) then
        # UTF-8 encoding. Non-ASCII characters must be encoded, not truncated.
        out = bytearray()
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\\\' and i + 1 < len(s):
                n = s[i + 1]
                if n == 'x':
                    out.append(int(s[i + 2:i + 4], 16))
                    i += 4
                    continue
                m = {'t': 9, 'n': 10, 'r': 13}
                if n in m:
                    out.append(m[n])
                else:
                    out.extend(n.encode('utf-8'))
                i += 2
                continue
            out.extend(c.encode('utf-8'))
            i += 1
        return bytes(out)

    n_vocab = hp['n_vocab']
    tokens = [None] * n_vocab
    scores = [0.0] * n_vocab
    types = [5] * n_vocab
    # id 0: RWKV7's mandatory leading token, excluded from the txt file
    tokens[0] = '<s>'
    scores[0] = 0.0
    types[0] = 3  # CONTROL
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            sp = line.index(' ')
            tid = int(line[:sp])
            q = line[sp + 1]
            if q == 'b':
                end = line.rindex("' ")
                tok = line[sp + 3:end]
            else:
                assert q in ("'", '"'), f'vocab line {tid}: bad quote'
                end = line.rindex(q + ' ')
                tok = line[sp + 2:end]
            sc = float(line[end + 2:])
            gid = tid  # the demo tokenizer feeds raw txt ids (1-based) to the model
            assert 0 <= gid < n_vocab
            tokens[gid] = tok
            scores[gid] = sc
            types[gid] = 1
    # deduplicate: same decoded bytes -> keep the HIGHEST id. The reference
    # tokenizer builds its trie in reversed id order, so for the ~10.9k
    # duplicate byte sequences in the World vocab it always emits the
    # highest id; the lower-id duplicates are effectively untrained rows.
    seen = {}
    n_real = 0
    for gid in range(n_vocab - 1, -1, -1):
        if types[gid] != 1:
            continue
        n_real += 1
        b = unescape(tokens[gid])
        if b in seen:
            tokens[gid] = f'[UNUSED_{gid}]'
            types[gid] = 5
        else:
            seen[b] = gid
    n_ph = 0
    for gid in range(n_vocab):
        if tokens[gid] is None:
            tokens[gid] = f'[UNUSED_{gid}]'
            n_ph += 1
        elif types[gid] == 5:
            n_ph += 1
    print(f'vocab: {n_vocab - n_ph} real entries, {n_ph} placeholders', flush=True)

    # ---- header + tensor infos (deterministic) ----
    hb = io.BytesIO()
    hb.write(struct.pack('<I', GGUF_MAGIC))
    hb.write(struct.pack('<I', GGUF_VERSION))
    hb.write(struct.pack('<Q', n_tensors))
    hb.write(struct.pack('<Q', 25))
    kv_str(hb, 'general.architecture', 'rwkv7')
    kv_str(hb, 'general.name', 'RWKV7-G1k-13.3B-20260930')
    kv_u32(hb, 'general.file_type', 1)
    kv_u32(hb, 'rwkv7.wkv.head_size', 64)
    kv_u32(hb, 'rwkv7.attention.decay_lora_rank', hp['rank_decay'])
    kv_u32(hb, 'rwkv7.attention.iclr_lora_rank', hp['rank_iclr'])
    kv_u32(hb, 'rwkv7.attention.value_residual_mix_lora_rank', hp['rank_v'])
    kv_u32(hb, 'rwkv7.attention.gate_lora_rank', hp['rank_gate'])
    kv_u32(hb, 'rwkv7.token_shift_count', 2)
    kv_u32(hb, 'rwkv7.context_length', 25600)
    kv_u32(hb, 'rwkv7.embedding_length', hp['n_embd'])
    kv_u32(hb, 'rwkv7.block_count', hp['n_layer'])
    kv_u32(hb, 'rwkv7.feed_forward_length', hp['n_ff'])
    kv_f32(hb, 'rwkv7.attention.layer_norm_epsilon', 1e-5)
    kv_u32(hb, 'rwkv7.vocab_size', n_vocab)
    kv_str(hb, 'tokenizer.ggml.model', 'rwkv')
    kv_str(hb, 'tokenizer.ggml.pre', 'rwkv')
    kv_arr_str(hb, 'tokenizer.ggml.tokens', tokens)
    kv_arr_f32(hb, 'tokenizer.ggml.scores', scores)
    kv_arr_i32(hb, 'tokenizer.ggml.token_type', types)
    kv_u32(hb, 'tokenizer.ggml.bos_token_id', 0)
    kv_u32(hb, 'tokenizer.ggml.eos_token_id', 0)
    kv_u32(hb, 'tokenizer.ggml.pad_token_id', 0)
    kv_bool(hb, 'tokenizer.ggml.add_bos_token', True)
    kv_bool(hb, 'tokenizer.ggml.add_eos_token', False)
    # id 0 is RWKV7's mandatory leading token (the reference demo prepends
    # [0] to every prompt); llama.cpp adds it via the BOS setting above
    header_bytes = hb.getvalue()

    ib = io.BytesIO()
    off = 0
    offsets = []
    for name, ne, gtype, pk, arr, zeros in plan:
        w_str(ib, name)
        ib.write(struct.pack('<I', len(ne)))
        for d in ne:
            ib.write(struct.pack('<Q', d))
        ib.write(struct.pack('<I', gtype))
        ib.write(struct.pack('<Q', off))
        nel = 1
        for d in ne:
            nel *= d
        size = (nel // 32) * 46 if gtype == GGML_BF16X else nel * 4
        offsets.append(off)
        off += size
        off += (-off) % GGUF_ALIGN
    D = (len(header_bytes) + len(ib.getvalue()) + GGUF_ALIGN - 1) // GGUF_ALIGN * GGUF_ALIGN

    state_file = out_path + '.state'
    if mode == 'meta':
        with open(out_path, 'wb') as f:
            f.write(header_bytes)
            f.write(ib.getvalue())
            f.write(b'\0' * (D - f.tell()))
        with open(state_file, 'w') as f:
            f.write('0')
        print(f'meta written: D={D}, tensors={n_tensors}. now run: data 0 {n_tensors}', flush=True)
        return

    # ---- data pass [first, last) with resume from the state file ----
    # LoRA matrices (w1/w2/a1/a2/v1/v2/g1/g2) are RIGHT-multiply matrices:
    # torch stores them (C, rank)/(rank, C) row-major, while the ggml tensor
    # (ne0-fastest) needs the transposed byte order. nn.Linear weights
    # ((out, in) torch) already match ggml's (in, out) layout directly.
    TRANS = {'time_mix_w1', 'time_mix_w2', 'time_mix_a1', 'time_mix_a2',
             'time_mix_v1', 'time_mix_v2', 'time_mix_g1', 'time_mix_g2'}
    with open(state_file) as f:
        done = int(f.read().strip())
    a = max(first, done)
    b = min(last, n_tensors)
    print(f'writing tensors [{a}, {b}) of {n_tensors}, D={D}', flush=True)

    def produce(entry):
        name, ne, gtype, pth_key, arr, zeros = entry
        def maybe_transpose(x):
            parts = name.split('.')
            short = '.'.join(parts[2:])
            if short.endswith('.weight'):
                short = short[:-len('.weight')]
            if short in TRANS:
                return np.ascontiguousarray(x.T)
            return x
        if gtype == GGML_BF16X:
            if zeros:
                x = np.zeros((int(np.prod(ne)),), dtype=np.float32)
            elif pth_key and isinstance(pth_key, str):
                x = maybe_transpose(sd[pth_key].detach().to(torch.float32).numpy())
            else:  # fused lerp
                x = lerp_fused_np(sd, pth_key)
            return quantize_tensor(np.ascontiguousarray(x)).reshape(-1).tobytes()
        if zeros:
            return b'\0' * (int(np.prod(ne)) * 4)
        if pth_key and isinstance(pth_key, str):
            x = maybe_transpose(sd[pth_key].detach().to(torch.float32).numpy())
            return np.ascontiguousarray(x, dtype='<f4').tobytes()
        x = lerp_fused_np(sd, pth_key)
        return np.ascontiguousarray(x, dtype='<f4').tobytes()

    with open(out_path, 'r+b') as f:
        for idx in range(a, b):
            name, ne, gtype = plan[idx][0], plan[idx][1], plan[idx][2]
            f.seek(D + offsets[idx])
            data = produce(plan[idx])
            f.write(data)
            with open(state_file, 'w') as f2:
                f2.write(str(idx + 1))
            print(f'  [{idx + 1}/{n_tensors}] {name} '
                  f'{"bf16x" if plan[idx][2] == GGML_BF16X else "f32"} done', flush=True)
    print(f'data pass complete: [{a}, {b})', flush=True)

if __name__ == '__main__':
    main()
