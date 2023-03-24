import argparse

import numpy as np
from sentencepiece import SentencePieceProcessor

from . import open_trace

def parse_args():
    parser = argparse.ArgumentParser(description='Upgrade old ggml model files to the current format')
    parser.add_argument('trace_file', help='tracefile to read')
    parser.add_argument('--tokenizer', help='path to LLaMA tokenizer.model file',
        dest='tokenizer_model_file', default='models/tokenizer.model')
    parser.add_argument('--temp', help='Sampling temperature',
        dest='temperature', default=0.8, type=float)
    parser.add_argument('--top_k', help='top k tokens to sample', type=int)
    parser.add_argument('--top_p', help='nucleus probability', type=float, default=1.0)
    return parser.parse_args()


def top_k_indices(logits, k):
    idxs = np.argpartition(logits, -k)[-k:]
    idxs = idxs[np.argsort(logits[idxs])][::-1]
    return idxs

def process_logits(logits, temp):
    logits = logits / temp
    logp = logits - logits.max()
    p = np.exp(logp)
    sum_p = p.sum()
    entropy = -(p * logp).sum() / sum_p + np.log(sum_p)
    p /= sum_p
    #entropy = -(p * np.log(p)).sum()
    return p, entropy

def top_p(p, top_p):
    if top_p < 1:
        cumsum = 0.
        for i in range(len(p)):
            cumsum += p[i]
            if cumsum >= top_p:
                return i + 1
    return len(p)

def replicate_sampler(tokens, args, max_print=10):
    log2 = np.log(2)
    tokenizer = SentencePieceProcessor(args.tokenizer_model_file)
    piece_repr = lambda tokid: repr(tokenizer.id_to_piece(int(tokid)))
    for tokens, logits_arrs in f:
        for tokid, logits in zip(tokens, logits_arrs):
            idxs = None
            if args.top_k is not None:
                idxs = top_k_indices(logits, args.top_k)
            else:
                idxs = np.argsort(logits)[::-1]
            logits = logits[idxs]
            p, entropy = process_logits(logits, args.temperature)

            n_top_p = top_p(p, args.top_p)
            logits = logits[:n_top_p]
            idxs = idxs[:n_top_p]

            print(f'in:{piece_repr(tokid):10} logits: mean={logits.mean()=:5.2f} max={logits[0]:5.2f} entropy={entropy*log2:.2f} bits n={len(idxs)}')
            print(' '*13, ' '.join(f'{piece_repr(candtok)}:{prob:.2f}' for candtok, prob in zip(idxs[:max_print], p)))

if __name__ == "__main__":
    args = parse_args()

    with open_trace(args.trace_file) as f:
        print(f'n_vocab={f.n_vocab}')
        replicate_sampler(f, args)