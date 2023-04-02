import numpy as np
import torch
from typing import Dict
from torch.nn import functional as F

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8, logit_bias: Dict[int, float] = None) -> int:
    probs = F.softmax(out.cpu(), dim=-1).numpy()

    return sample_probs(probs, temperature, top_p, logit_bias)

def sample_probs(probs: np.ndarray, temperature: float = 1.0, top_p: float = 0.8, logit_bias: Dict[int, float] = None) -> int:
    assert 0.0 <= temperature, 'temperature'
    assert 0.0 <= top_p <= 1.0, 'top_p'

    if top_p == 0.0:
        top_p = 1.0

    if logit_bias is not None:
        logits = np.log(probs)

        for token in logit_bias.keys():
            logits[token] += logit_bias[token]

        probs = np.exp(logits) / np.sum(np.exp(logits))

    if temperature == 0.0:
        return np.argmax(probs).item()

    if top_p < 1.0:
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0

    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)

    probs = probs / np.sum(probs)

    return np.random.choice(a=len(probs), p=probs)
