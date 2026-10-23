# Reference RWKV7 forward pass in numpy (official math from BlinkDL/RWKV-LM
# rwkv_v7_numpy.py by johanwind), used to validate GGUF conversions.
#
#   python compare_rwkv7_ref.py <model.safetensors> <vocab.txt> "<prompt>"
#
# Prints the token ids and the top-10 next-token distribution, which should
# match `llama-cli`/`llama-tokenize` output from the converted GGUF.
import sys

import numpy as np
from safetensors.torch import load_file

MODEL = sys.argv[1] if len(sys.argv) > 1 else r'F:\rwkv\models\rwkv7-0.1b-hf\model.safetensors'
VOCAB = sys.argv[2] if len(sys.argv) > 2 else r'F:\rwkv\RWKV-LM\RWKV-v7\rwkv_vocab_v20230424.txt'
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "The Eiffel tower is in the city of"
USE_BOS = True  # the reference tokenizer always prepends token 0

layer_norm = lambda x, w, b: (x - x.mean()) / (x.var() + 1e-5)**0.5 * w + b
group_norm = lambda x, w, b: ((x - x.mean(axis=1, keepdims=1)) /
                              (x.var(axis=1, keepdims=1) + 64e-5)**0.5).flatten() * w + b
sigmoid = lambda x: 1 / (1 + np.exp(-x))

raw = load_file(MODEL)
W = {(k[6:] if k.startswith('rwkv7.') else k): v.squeeze().float().numpy()
     for k, v in raw.items()}

n_layer = 1 + max(int(k.split('.')[1]) for k in W if k.startswith('blocks.'))
n_embd = W['emb.weight'].shape[1]
n_head = n_embd // 64
head_size = 64


def time_mixing(x, v0, last_x, S, i):
    p = f'blocks.{i}.att.'
    mr, mw, mk, mv, ma, mg = [W[p + 'x_' + c] for c in 'rwkvag']
    w_bias = W[p + 'w0'].reshape(-1)
    r_k = W[p + 'r_k']
    Ww1, Ww2 = W[p + 'w1'], W[p + 'w2']
    Wa1, Wa2, a_bias = W[p + 'a1'], W[p + 'a2'], W[p + 'a0'].reshape(-1)
    Wg1, Wg2 = W[p + 'g1'], W[p + 'g2']
    k_k, k_a = W[p + 'k_k'].reshape(-1), W[p + 'k_a'].reshape(-1)
    Wr, Wk, Wv, Wo = (W[p + 'receptance.weight'], W[p + 'key.weight'],
                      W[p + 'value.weight'], W[p + 'output.weight'])
    ln_w, ln_b = W[p + 'ln_x.weight'], W[p + 'ln_x.bias']

    xr, xw, xk, xv, xa, xg = [x + m * (last_x - x) for m in [mr, mw, mk, mv, ma, mg]]
    r = Wr @ xr
    w = np.exp(-sigmoid(np.tanh(xw @ Ww1) @ Ww2 + w_bias) / np.e**0.5)
    k = Wk @ xk
    v = Wv @ xv
    if v0 is None:
        v0 = v
    else:
        Wv2, Wv1, v_bias = W[p + 'v2'], W[p + 'v1'], W[p + 'v0'].reshape(-1)
        v += (v0 - v) * sigmoid(xv @ Wv1 @ Wv2 + v_bias)
    a = sigmoid(xa @ Wa1 @ Wa2 + a_bias)
    g = sigmoid(xg @ Wg1) @ Wg2
    kk = k * k_k
    k = k + k * (a - 1) * k_a

    r, w, k, v, kk, a, r_k = [i_.reshape(n_head, head_size, 1)
                              for i_ in [r, w, k, v, kk, a, r_k]]
    kk = kk / np.maximum(np.linalg.norm(kk, axis=1, keepdims=1), 1e-12)

    S = S * w.mT - S @ kk * (kk * a).mT + v * k.mT
    y = S @ r
    y = group_norm(y, ln_w, ln_b)
    y = y + ((r * k * r_k).sum(axis=1, keepdims=1) * v).flatten()
    return Wo @ (y * g), v0, x, S


def channel_mixing(x, last_x, mix, Wk, Wv):
    k = Wk @ (x + mix * (last_x - x))
    v = Wv @ np.maximum(k, 0)**2
    return v, x


def rwkv7(token, state):
    x = W['emb.weight'][token]
    x = layer_norm(x, W['blocks.0.ln0.weight'], W['blocks.0.ln0.bias'])
    v0 = None
    for i in range(n_layer):
        x_ = layer_norm(x, W[f'blocks.{i}.ln1.weight'], W[f'blocks.{i}.ln1.bias'])
        dx, v0, state[0][i, 0], state[1][i] = time_mixing(
            x_, v0, state[0][i, 0], state[1][i], i)
        x = x + dx
        x_ = layer_norm(x, W[f'blocks.{i}.ln2.weight'], W[f'blocks.{i}.ln2.bias'])
        dx, state[0][i, 1] = channel_mixing(
            x_, state[0][i, 1], W[f'blocks.{i}.ffn.x_k'].reshape(-1),
            W[f'blocks.{i}.ffn.key.weight'], W[f'blocks.{i}.ffn.value.weight'])
        x = x + dx
    x = layer_norm(x, W['ln_out.weight'], W['ln_out.bias'])
    return W['head.weight'] @ x, state


class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.i2t = {}
        lines = open(file_name, encoding='utf-8').readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode('utf-8') if isinstance(x, str) else x
            self.i2t[idx] = x
        self.t2i = {v: k for k, v in self.i2t.items()}
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0] * 256
        for idx, s in reversed(sorted(self.i2t.items())):
            if len(s) >= 2:
                s0, s1 = s[0], s[1]
                self.table[s0][s1].append(s)
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encode(self, text):
        src = text.encode('utf-8')
        out = []
        i = 0
        while i < len(src):
            s = src[i:i + 1]
            if i < len(src) - 1 and src[i + 1] in self.good[src[i]]:
                cand = src[i:i + self.wlen[src[i]]]
                for t in self.table[src[i]][src[i + 1]]:
                    if cand.startswith(t):
                        s = t
                        break
            out.append(self.t2i[s])
            i += len(s)
        return out

    def decode(self, ids):
        return b''.join(self.i2t.get(i, b'?') for i in ids).decode('utf-8', 'replace')


tok = RWKV_TOKENIZER(VOCAB)
ids = ([0] if USE_BOS else []) + tok.encode(PROMPT)
print('prompt ids:', ids)

state = (np.zeros((n_layer, 2, n_embd), dtype=np.float32),
         np.zeros((n_layer, n_head, head_size, head_size), dtype=np.float32))
for t in ids:
    logits, state = rwkv7(t, state)

probs = np.exp(logits - logits.max())
probs /= probs.sum()
order = np.argsort(-probs)[:10]
print('top-10:')
for i in order:
    print(f'  id {i} {tok.decode([i])!r} p={probs[i]:.4f}')
