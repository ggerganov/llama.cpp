import sys, struct, math, argparse

import numpy as np

import gguf

# Note: Does not support GGML_QKK_64
QK_K = 256
# Items here are (block size, type size)
GGML_QUANT_SIZES = {
    gguf.GGMLQuantizationType.F32  : (1, 4),
    gguf.GGMLQuantizationType.F16  : (1, 2),
    gguf.GGMLQuantizationType.Q4_0 : (32, 2 + 16),
    gguf.GGMLQuantizationType.Q4_1 : (32, 2 + 2 + 16),
    gguf.GGMLQuantizationType.Q5_0 : (32, 2 + 4 + 16),
    gguf.GGMLQuantizationType.Q5_1 : (32, 2 + 2 + 4 + 16),
    gguf.GGMLQuantizationType.Q8_0 : (32, 2 + 32),
    gguf.GGMLQuantizationType.Q8_1 : (32, 4 + 4 + 32),
    gguf.GGMLQuantizationType.Q2_K : (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    gguf.GGMLQuantizationType.Q3_K : (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    gguf.GGMLQuantizationType.Q4_K : (256, 2 + 2 + QK_K // 2 + 12),
    gguf.GGMLQuantizationType.Q5_K : (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    gguf.GGMLQuantizationType.Q6_K : (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    gguf.GGMLQuantizationType.Q8_K : (256, 2 + QK_K + QK_K // 8),
}

class Hyperparameters:
    def __init__(self):
        self.n_vocab = self.n_embd = self.n_mult = self.n_head = self.n_layer = self.n_rot = self.ftype = 0

    def load(self, data, offset):
        (
            self.n_vocab,
            self.n_embd,
            self.n_mult,
            self.n_head,
            self.n_layer,
            self.n_rot,
            self.ftype,
        ) = struct.unpack('<7I', data[offset:offset + (4 * 7)])
        return 4 * 7

    def __str__(self):
        return f'<Hyperparameters: n_vocab={self.n_vocab}, n_embd={self.n_embd}, n_mult={self.n_mult}, n_head={self.n_head}, n_layer={self.n_layer}, n_rot={self.n_rot}, ftype={self.ftype}>'

class Vocab:
    def __init__(self):
        self.items = []

    def load(self, data, offset, n_vocab):
        orig_offset = offset
        for _ in range(n_vocab):
            itemlen = struct.unpack('<I', data[offset:offset + 4])[0]
            assert itemlen < 4096, 'Absurd vocab item length'
            offset += 4
            vocab = bytes(data[offset:offset + itemlen])
            offset += itemlen
            score = struct.unpack('<f', data[offset:offset + 4])[0]
            offset += 4
            self.items.append((vocab, score))
        return offset - orig_offset

class Tensor:
    def __init__(self):
        self.name = None
        self.dims = ()
        self.dtype = None
        self.start_offset = 0
        self.len_bytes = 0

    def load(self, data, offset):
        orig_offset = offset
        (n_dims, name_len, dtype) = struct.unpack('<3I', data[offset:offset + 12])
        assert n_dims >= 0 and n_dims <= 4, f'Invalid tensor dimensions {n_dims}'
        assert name_len < 4096, 'Absurd tensor name length'
        quant = GGML_QUANT_SIZES.get(dtype)
        assert quant is not None, 'Unknown tensor type'
        (blksize, tysize) = quant
        offset += 12
        self.dtype= dtype
        self.dims = struct.unpack(f'<{n_dims}I', data[offset:offset + (4 * n_dims)])
        offset += 4 * n_dims
        self.name = bytes(data[offset:offset + name_len])
        offset += name_len
        pad = ((offset + 31) & ~31) - offset
        offset += pad
        n_elems = np.prod(self.dims)
        n_bytes = (n_elems * tysize) // blksize
        self.start_offset = offset
        self.len_bytes = n_bytes
        offset += n_bytes
        # print(n_dims, name_len, dtype, self.dims, self.name, pad)
        return offset - orig_offset

class GGMLV3Model:
    def __init__(self):
        self.hyperparameters = None
        self.vocab = None
        self.tensor_map = {}
        self.tensors = []

    def validate_header(self, data, offset):
        if bytes(data[offset:offset + 4]) != b'tjgg' or struct.unpack('<I', data[offset + 4:offset + 8])[0] != 3:
            raise ValueError('Only GGJTv3 supported')
        return 8

    def load(self, data, offset):
        offset += self.validate_header(data, offset)
        hp = Hyperparameters()
        offset += hp.load(data, offset)
        vocab = Vocab()
        offset += vocab.load(data, offset, hp.n_vocab)
        tensors = []
        tensor_map = {}
        while offset < len(data):
            tensor = Tensor()
            offset += tensor.load(data, offset)
            tensor_map[tensor.name] = len(tensors)
            tensors.append(tensor)
        self.hyperparameters = hp
        self.vocab = vocab
        self.tensors = tensors
        self.tensor_map = tensor_map
        return offset

def save_gguf(ggml_model, data, cfg):
    hp = ggml_model.hyperparameters
    ff_tensor_idx = ggml_model.tensor_map.get(b'layers.0.feed_forward.w1.weight')
    assert ff_tensor_idx is not None, 'Missing layer 0 FF tensor'
    ff_tensor = ggml_model.tensors[ff_tensor_idx]
    if cfg.gqa == 1:
        n_kv_head = hp.n_head
    else:
        gqa = float(cfg.gqa)
        n_kv_head = None
        for x in range(1, 256):
            if float(hp.n_head) / float(x) == gqa:
                n_kv_head = x
        assert n_kv_head is not None, "Couldn't determine n_kv_head from GQA param"
        print(f'- Guessed n_kv_head = {n_kv_head} based on GQA {cfg.gqa}')
    nm = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, hp.n_layer)
    gguf_writer = gguf.GGUFWriter(cfg.output, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.LLAMA], use_temp_file = False)
    #gguf_writer.add_name('meep')
    #gguf_writer.add_source_hf_repo('merp')
    # gguf_writer.add_tensor_data_layout("Meta AI original pth")
    gguf_writer.add_context_length(cfg.context_length)
    gguf_writer.add_embedding_length(hp.n_embd)
    gguf_writer.add_block_count(hp.n_layer)
    gguf_writer.add_feed_forward_length(ff_tensor.dims[1])
    print('FF dim', ff_tensor.dims[1])
    gguf_writer.add_rope_dimension_count(hp.n_embd // hp.n_head)
    gguf_writer.add_head_count(hp.n_head)
    gguf_writer.add_head_count_kv(n_kv_head)
    gguf_writer.add_layer_norm_rms_eps(float(cfg.eps))
    gguf_writer.add_tokenizer_model('llama')
    tokens = []
    scores = []
    print(f'* Adding {hp.n_vocab} vocab item(s)')
    toktypes = []
    for (tokid, (vbytes, vscore)) in enumerate(ggml_model.vocab.items):
        if len(vbytes) > 1 and vbytes[0] == 32:
            vbytes = vbytes.replace(b' ', b'\xe2\x96\x81')
        tt = 1
        if len(vbytes) == 0:
            tt = 3
        elif tokid >= 3 and tokid <= 258 and len(vbytes) == 1:
            hv = hex(vbytes[0])[2:].upper()
            vbytes = bytes(f'<0x{hv}>', encoding = 'UTF-8')
            tt = 6
        toktypes.append(tt)
        tokens.append(vbytes)
        scores.append(vscore)
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)
    print('* Adding tensors')
    for tensor in ggml_model.tensors:
        name = str(tensor.name, 'UTF-8')
        if name.endswith('.weight'):
            name = name[:-7]
            suffix = '.weight'
        elif name.endswith('.bias'):
            name = name[:-5]
            suffix = '.bias'
        mapped_name = nm.get(name)
        assert mapped_name is not None, f'Bad name {name}'
        mapped_name += suffix
        tempdims = list(tensor.dims[:])
        if len(tempdims) > 1:
            temp = tempdims[1]
            tempdims[1] = tempdims[0]
            tempdims[0] = temp
        print(f'+ {tensor.name} | {mapped_name} {tensor.dims} :: {tempdims}')
        gguf_writer.add_tensor(mapped_name, data[tensor.start_offset:tensor.start_offset + tensor.len_bytes], raw_shape = tempdims, raw_dtype = tensor.dtype)
    print("gguf: write header")
    gguf_writer.write_header_to_file()
    print("gguf: write metadata")
    gguf_writer.write_kv_data_to_file()
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

def handle_args():
    parser = argparse.ArgumentParser(description = 'Convert GGMLv3 models to GGUF')
    parser.add_argument('--input', '-i', help = 'Input GGMLv3 filename')
    parser.add_argument('--output', '-o', help ='Output GGUF filename')
    parser.add_argument('--gqa', type = int, default = 1, help = 'grouped-query attention factor (use 8 for LLaMA2 70B)')
    parser.add_argument('--eps', default = '5.0e-06', help = 'RMS norm eps (use 1e-5 for LLaMA2)')
    parser.add_argument('--context-length', '-c', type=int, default = 2048, help = 'Default max context length')
    return parser.parse_args()

def main():
    cfg = handle_args()
    data = np.memmap(cfg.input, mode = 'r')
    model = GGMLV3Model()
    offset = model.load(data, 0)
    print(model.hyperparameters)
    # print(model.vocab.items)
    # return
    save_gguf(model, data, cfg)

main()
