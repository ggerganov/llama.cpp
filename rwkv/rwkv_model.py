# Reference implementation of RWKV in PyTorch.

# Original code: https://github.com/BlinkDL/ChatRWKV/blob/0d0abf181356c6f27501274cad18bdf28c83a45b/RWKV_in_150_lines.py
# Original code by https://github.com/BlinkDL, licensed under Apache License 2.0

# Improvements made to the original code:
# - safetensors loading support
# - LoRA loading support
# - ln0 absortion support
# - general code style improvements

import time
import torch
import types
from typing import Union, Tuple, Dict, Optional
from torch.nn import functional as F

LORA_R: int = 4
LORA_ALPHA: int = 32

def load_state_dict(file_path: str, device: str) -> Dict[str, torch.Tensor]:
    print(f'Loading {file_path}')

    if file_path.endswith('.safetensors'):
        from safetensors import safe_open

        w = {}

        with safe_open(file_path, framework='pt', device=device) as state_dict:
            for key in state_dict.keys():
                w[key] = state_dict.get_tensor(key)

        return w
    else:
        return torch.load(file_path, map_location=device)

def get_layer_count(state_dict: Dict[str, torch.Tensor]) -> int:
    n_layer = 0

    while f'blocks.{n_layer}.ln1.weight' in state_dict:
        n_layer += 1

    assert n_layer > 0

    return n_layer

class RWKV_RNN(torch.jit.ScriptModule):

    def __init__(
            self,
            model_path: str,
            additional_model_path: Optional[str] = None,
            device: str = 'cpu',
            absorb_layer_norm_0: bool = False
    ):
        super().__init__()

        self.representation: torch.Tensor = torch.tensor([0], dtype=torch.float32, device=device)
        self.eval()

        print(f'Loading RWKV model from {model_path}')

        w = load_state_dict(model_path, device)

        if additional_model_path is not None:
            additional_w = load_state_dict(additional_model_path, device)

            for k in additional_w:
                if k != '_training_state':
                    w[k] = additional_w[k]

        print('Merging LoRA into weights')

        start = time.time()

        for k in list(w.keys()):
            module_k = k.replace('.weight', '')

            if module_k + '.lora_A.weight' in w:
                lora_A = w[module_k + '.lora_A.weight']
                lora_B = w[module_k + '.lora_B.weight']
                assert lora_B.shape[1] == lora_A.shape[0] == LORA_R
                w[module_k + '.weight'] = w[module_k + '.weight'] + lora_B @ lora_A * (LORA_ALPHA / LORA_R)
                del w[module_k + '.lora_A.weight']
                del w[module_k + '.lora_B.weight']
                del lora_A
                del lora_B

        print('Took %.3f sec' % ((time.time() - start),))

        for k in w.keys():
            if '.time_' in k:
                # (1, 1, n_embed) -> (n_embed)
                w[k] = w[k].squeeze()

            if '.time_decay' in k:
                # The real time decay is like e^{-e^x}
                w[k] = -torch.exp(w[k].float())
            elif w[k].dtype != torch.float32:
                w[k] = w[k].float()

        self.w = types.SimpleNamespace()
        self.w.blocks = {}

        # Example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
        for k in w.keys():
            parts = k.split('.')
            last = parts.pop()
            here = self.w

            for p in parts:
                if p.isdigit():
                    p = int(p)

                    if p not in here:
                        here[p] = types.SimpleNamespace()

                    here = here[p]
                else:
                    if not hasattr(here, p):
                        setattr(here, p, types.SimpleNamespace())

                    here = getattr(here, p)

            setattr(here, last, w[k])

        self.absorb_layer_norm_0 = absorb_layer_norm_0

        if absorb_layer_norm_0:
            print('Absorbing first LayerNorm into embedding matrix')

            start = time.time()

            for i in range(len(self.w.emb.weight)):
                self.w.emb.weight[i] = self.layer_norm(self.w.emb.weight[i], self.w.blocks[0].ln0)

            print('Took %.3f sec' % ((time.time() - start),))

        self.n_layer = get_layer_count(w)
        self.n_embed = self.w.emb.weight.shape[1]

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.n_embed,), weight=w.weight, bias=w.bias)

    @torch.jit.script_method
    def channel_mixing(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        xk = x * time_mix_k + state[5 * i + 0] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5 * i + 0] * (1 - time_mix_r)
        state[5 * i + 0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))  # square relu, primer paper
        return r * (vw @ k)

    @torch.jit.script_method
    def time_mixing(self, x, state, i: int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xk = x * time_mix_k + state[5 * i + 1] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5 * i + 1] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5 * i + 1] * (1 - time_mix_r)
        state[5 * i + 1] = x
        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        aa = state[5 * i + 2]
        bb = state[5 * i + 3]
        pp = state[5 * i + 4]
        ww = time_first + k
        qq = torch.maximum(pp, ww)
        e1 = torch.exp(pp - qq)
        e2 = torch.exp(ww - qq)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        wkv = a / b
        ww = pp + time_decay
        qq = torch.maximum(ww, k)
        e1 = torch.exp(ww - qq)
        e2 = torch.exp(k - qq)
        state[5 * i + 2] = e1 * aa + e2 * v
        state[5 * i + 3] = e1 * bb + e2
        state[5 * i + 4] = qq
        return ow @ (r * wkv)

    def warm_up(self):
        print('Warming up the model')
        start = time.time()
        self.forward(0, None)
        print('Took %.3f sec' % ((time.time() - start),))

    def forward(self, token: int, state: Union[torch.Tensor, None], save_representation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x: torch.Tensor = self.w.emb.weight[token]

            if state is None:
                state = torch.zeros(self.n_layer * 5, self.n_embed, device=x.device)

                for i in range(self.n_layer):
                    # ~Negative infinity
                    state[5 * i + 4] = -1e30

            if not self.absorb_layer_norm_0:
                x = self.layer_norm(x, self.w.blocks[0].ln0)

            for i in range(self.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln1),
                    state,
                    i,
                    att.time_mix_k,
                    att.time_mix_v,
                    att.time_mix_r,
                    att.time_first,
                    att.time_decay,
                    att.key.weight,
                    att.value.weight,
                    att.receptance.weight,
                    att.output.weight
                )

                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(
                    self.layer_norm(x, self.w.blocks[i].ln2),
                    state,
                    i,
                    ffn.time_mix_k,
                    ffn.time_mix_r,
                    ffn.key.weight,
                    ffn.value.weight,
                    ffn.receptance.weight
                )

            x = self.layer_norm(x, self.w.ln_out)

            if save_representation:
                self.representation = x.clone()

            x = (self.w.head.weight @ x).float()

            return x, state
