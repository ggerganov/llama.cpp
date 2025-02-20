from __future__ import annotations

from typing import Sequence

from .constants import MODEL_ARCH, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES


class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",                         # gptneox
            "transformer.wte",                           # gpt2 gpt-j mpt refact qwen dbrx jais exaone
            "transformer.word_embeddings",               # falcon
            "word_embeddings",                           # bloom
            "model.embed_tokens",                        # llama-hf nemotron olmoe olmo2 rwkv6qwen2
            "tok_embeddings",                            # llama-pth
            "embeddings.word_embeddings",                # bert nomic-bert
            "language_model.embedding.word_embeddings",  # persimmon
            "wte",                                       # gpt2
            "transformer.embd.wte",                      # phi2
            "model.tok_embeddings",                      # internlm2
            "model.embedding",                           # mamba-qbert
            "backbone.embedding",                        # mamba
            "backbone.embeddings",                       # mamba-hf
            "transformer.in_out_embed",                  # Grok
            "embedding.word_embeddings",                 # chatglm
            "transformer.token_embeddings",              # openelm
            "shared",                                    # t5
            "rwkv.embeddings",                           # rwkv
        ),

        # Token type embeddings
        MODEL_TENSOR.TOKEN_TYPES: (
            "embeddings.token_type_embeddings",  # bert nomic-bert
        ),

        # Normalization of token embeddings
        MODEL_TENSOR.TOKEN_EMBD_NORM: (
            "word_embeddings_layernorm",  # bloom
            "embeddings.LayerNorm",       # bert
            "emb_ln",                     # nomic-bert
            "transformer.norm",           # openelm
            "rwkv.blocks.0.pre_ln",       # rwkv
            "backbone.norm",              # wavtokenizer
        ),

        # Position embeddings
        MODEL_TENSOR.POS_EMBD: (
            "transformer.wpe",                 # gpt2
            "embeddings.position_embeddings",  # bert
            "wpe",                             # gpt2
        ),

        # Output
        MODEL_TENSOR.OUTPUT: (
            "embed_out",                 # gptneox
            "lm_head",                   # gpt2 mpt falcon llama-hf baichuan qwen mamba dbrx jais nemotron exaone olmoe olmo2 phimoe
            "output",                    # llama-pth bloom internlm2
            "word_embeddings_for_head",  # persimmon
            "lm_head.linear",            # phi2
            "output_layer",              # chatglm
            "head",                      # rwkv
            "head.out",                  # wavtokenizer
        ),

        # Output norm
        MODEL_TENSOR.OUTPUT_NORM: (
            "gpt_neox.final_layer_norm",               # gptneox
            "transformer.ln_f",                        # gpt2 gpt-j falcon jais exaone
            "model.norm",                              # llama-hf baichuan internlm2 olmoe olmo2 phimoe
            "norm",                                    # llama-pth
            "transformer.norm_f",                      # mpt dbrx
            "ln_f",                                    # refact bloom qwen gpt2
            "language_model.encoder.final_layernorm",  # persimmon
            "model.final_layernorm",                   # persimmon
            "lm_head.ln",                              # phi2
            "model.norm_f",                            # mamba-qbert
            "backbone.norm_f",                         # mamba
            "transformer.rms_norm",                    # Grok
            "encoder.final_layernorm",                 # chatglm
            "transformer.norm",                        # openelm
            "model.norm",                              # nemotron
            "rwkv.ln_out",                             # rwkv
            "backbone.final_layer_norm",               # wavtokenizer
        ),

        # Rope frequencies
        MODEL_TENSOR.ROPE_FREQS: (
            "rope.freqs",  # llama-pth
            "rotary_pos_emb.inv_freq",  # chatglm
        ),

        MODEL_TENSOR.ROPE_FACTORS_LONG: (),
        MODEL_TENSOR.ROPE_FACTORS_SHORT: (),

        MODEL_TENSOR.CONV1D: (
            "backbone.embed", # roberta
        ),
    }

    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",                # gptneox
            "transformer.h.{bid}.ln_1",                             # gpt2 gpt-j refact qwen jais exaone
            "transformer.blocks.{bid}.norm_1",                      # mpt
            "transformer.h.{bid}.input_layernorm",                  # falcon7b
            "h.{bid}.input_layernorm",                              # bloom
            "transformer.h.{bid}.ln_mlp",                           # falcon40b
            "model.layers.{bid}.input_layernorm",                   # llama-hf nemotron olmoe phimoe
            "layers.{bid}.attention_norm",                          # llama-pth
            "language_model.encoder.layers.{bid}.input_layernorm",  # persimmon
            "model.layers.{bid}.ln1",                               # yi
            "h.{bid}.ln_1",                                         # gpt2
            "transformer.h.{bid}.ln",                               # phi2
            "model.layers.layers.{bid}.norm",                       # plamo
            "model.layers.{bid}.attention_norm",                    # internlm2
            "model.layers.{bid}.norm",                              # mamba-qbert
            "backbone.layers.{bid}.norm",                           # mamba
            "transformer.decoder_layer.{bid}.rms_norm",             # Grok
            "transformer.blocks.{bid}.norm_attn_norm.norm_1",       # dbrx
            "encoder.layers.{bid}.input_layernorm",                 # chatglm
            "transformer.layers.{bid}.attn_norm",                   # openelm
            "rwkv.blocks.{bid}.ln1",                                # rwkv
        ),

        # Attention norm 2
        MODEL_TENSOR.ATTN_NORM_2: (
            "transformer.h.{bid}.ln_attn",                  # falcon40b
            "encoder.layer.{bid}.layer_norm_1",             # jina-v2-code
            "rwkv.blocks.{bid}.ln2",                        # rwkv
        ),

        # Attention query-key-value
        MODEL_TENSOR.ATTN_QKV: (
            "gpt_neox.layers.{bid}.attention.query_key_value",                     # gptneox
            "transformer.h.{bid}.attn.c_attn",                                     # gpt2 qwen jais
            "transformer.blocks.{bid}.attn.Wqkv",                                  # mpt
            "transformer.blocks.{bid}.norm_attn_norm.attn.Wqkv",                   # dbrx
            "transformer.h.{bid}.self_attention.query_key_value",                  # falcon
            "h.{bid}.self_attention.query_key_value",                              # bloom
            "language_model.encoder.layers.{bid}.self_attention.query_key_value",  # persimmon
            "model.layers.{bid}.self_attn.query_key_value",                        # persimmon
            "h.{bid}.attn.c_attn",                                                 # gpt2
            "transformer.h.{bid}.mixer.Wqkv",                                      # phi2
            "encoder.layers.{bid}.attn.Wqkv",                                      # nomic-bert
            "model.layers.{bid}.self_attn.qkv_proj",                               # phi3
            "encoder.layers.{bid}.self_attention.query_key_value",                 # chatglm
            "transformer.layers.{bid}.attn.qkv_proj",                              # openelm
        ),

        # Attention query
        MODEL_TENSOR.ATTN_Q: (
            "model.layers.{bid}.self_attn.q_proj",                       # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.self_attn.q_proj_no_perm",               # llama-custom
            "layers.{bid}.attention.wq",                                 # llama-pth
            "encoder.layer.{bid}.attention.self.query",                  # bert
            "transformer.h.{bid}.attn.q_proj",                           # gpt-j
            "model.layers.layers.{bid}.self_attn.q_proj",                # plamo
            "model.layers.{bid}.attention.wq",                           # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.query",# Grok
            "transformer.h.{bid}.attn.attention.q_proj",                 # exaone
        ),

        # Attention key
        MODEL_TENSOR.ATTN_K: (
            "model.layers.{bid}.self_attn.k_proj",                     # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.self_attn.k_proj_no_perm",             # llama-custom
            "layers.{bid}.attention.wk",                               # llama-pth
            "encoder.layer.{bid}.attention.self.key",                  # bert
            "transformer.h.{bid}.attn.k_proj",                         # gpt-j
            "transformer.h.{bid}.attn.k",                              # refact
            "model.layers.layers.{bid}.self_attn.k_proj",              # plamo
            "model.layers.{bid}.attention.wk",                         # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.key",# Grok
            "transformer.h.{bid}.attn.attention.k_proj",               # exaone
        ),

        # Attention value
        MODEL_TENSOR.ATTN_V: (
            "model.layers.{bid}.self_attn.v_proj",                       # llama-hf nemotron olmoe olmo2 phimoe
            "layers.{bid}.attention.wv",                                 # llama-pth
            "encoder.layer.{bid}.attention.self.value",                  # bert
            "transformer.h.{bid}.attn.v_proj",                           # gpt-j
            "transformer.h.{bid}.attn.v",                                # refact
            "model.layers.layers.{bid}.self_attn.v_proj",                # plamo
            "model.layers.{bid}.attention.wv",                           # internlm2
            "transformer.decoder_layer.{bid}.multi_head_attention.value",# Grok
            "transformer.h.{bid}.attn.attention.v_proj",                 # exaone
        ),

        # Attention output
        MODEL_TENSOR.ATTN_OUT: (
            "gpt_neox.layers.{bid}.attention.dense",                        # gptneox
            "transformer.h.{bid}.attn.c_proj",                              # gpt2 refact qwen jais
            "transformer.blocks.{bid}.attn.out_proj",                       # mpt
            "transformer.h.{bid}.self_attention.dense",                     # falcon
            "h.{bid}.self_attention.dense",                                 # bloom
            "model.layers.{bid}.self_attn.o_proj",                          # llama-hf nemotron olmoe olmo2 phimoe
            "model.layers.{bid}.self_attn.linear_attn",                     # deci
            "layers.{bid}.attention.wo",                                    # llama-pth
            "encoder.layer.{bid}.attention.output.dense",                   # bert
            "transformer.h.{bid}.attn.out_proj",                            # gpt-j
            "language_model.encoder.layers.{bid}.self_attention.dense",     # persimmon
            "model.layers.{bid}.self_attn.dense",                           # persimmon
            "h.{bid}.attn.c_proj",                                          # gpt2
            "transformer.h.{bid}.mixer.out_proj",                           # phi2
            "model.layers.layers.{bid}.self_attn.o_proj",                   # plamo
            "model.layers.{bid}.attention.wo",                              # internlm2
            "encoder.layers.{bid}.attn.out_proj",                           # nomic-bert
            "transformer.decoder_layer.{bid}.multi_head_attention.linear",  # Grok
            "transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",        # dbrx
            "encoder.layers.{bid}.self_attention.dense",                    # chatglm
            "transformer.layers.{bid}.attn.out_proj",                       # openelm
            "transformer.h.{bid}.attn.attention.out_proj",                  # exaone
        ),

        # Attention output norm
        MODEL_TENSOR.ATTN_OUT_NORM: (
            "encoder.layer.{bid}.attention.output.LayerNorm",  # bert
            "encoder.layers.{bid}.norm1",                      # nomic-bert
            "transformer.decoder_layer.{bid}.rms_norm_1",      # Grok
            "transformer.blocks.{bid}.norm_attn_norm.norm_2",  # dbrx
        ),

        MODEL_TENSOR.ATTN_POST_NORM: (
            "model.layers.{bid}.post_attention_layernorm",     # gemma2 olmo2
        ),

        # Rotary embeddings
        MODEL_TENSOR.ATTN_ROT_EMBD: (
            "model.layers.{bid}.self_attn.rotary_emb.inv_freq",        # llama-hf
            "layers.{bid}.attention.inner_attention.rope.freqs",       # llama-pth
            "model.layers.layers.{bid}.self_attn.rotary_emb.inv_freq", # plamo
            "transformer.h.{bid}.attn.rotary_emb.inv_freq",            # codeshell
        ),

        # Feed-forward norm
        MODEL_TENSOR.FFN_NORM: (
            "gpt_neox.layers.{bid}.post_attention_layernorm",                # gptneox
            "transformer.h.{bid}.ln_2",                                      # gpt2 refact qwen jais exaone
            "h.{bid}.post_attention_layernorm",                              # bloom
            "transformer.blocks.{bid}.norm_2",                               # mpt
            "model.layers.{bid}.post_attention_layernorm",                   # llama-hf nemotron olmoe phimoe
            "layers.{bid}.ffn_norm",                                         # llama-pth
            "language_model.encoder.layers.{bid}.post_attention_layernorm",  # persimmon
            "model.layers.{bid}.ln2",                                        # yi
            "h.{bid}.ln_2",                                                  # gpt2
            "model.layers.{bid}.ffn_norm",                                   # internlm2
            "transformer.decoder_layer.{bid}.rms_norm_2",                    # Grok
            "encoder.layers.{bid}.post_attention_layernorm",                 # chatglm
            "transformer.layers.{bid}.ffn_norm",                             # openelm
        ),

        # Post feed-forward norm
        MODEL_TENSOR.FFN_PRE_NORM: (
            "model.layers.{bid}.pre_feedforward_layernorm", # gemma2
        ),

        # Post feed-forward norm
        MODEL_TENSOR.FFN_POST_NORM: (
            "model.layers.{bid}.post_feedforward_layernorm", # gemma2 olmo2
        ),

        MODEL_TENSOR.FFN_GATE_INP: (
            "layers.{bid}.feed_forward.gate",                   # mixtral
            "model.layers.{bid}.block_sparse_moe.gate",         # mixtral phimoe
            "model.layers.{bid}.mlp.gate",                      # qwen2moe olmoe
            "transformer.decoder_layer.{bid}.router",           # Grok
            "transformer.blocks.{bid}.ffn.router.layer",        # dbrx
            "model.layers.{bid}.block_sparse_moe.router.layer", # granitemoe
        ),

        MODEL_TENSOR.FFN_GATE_INP_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert_gate", # qwen2moe
        ),

        MODEL_TENSOR.FFN_EXP_PROBS_B: (
            "model.layers.{bid}.mlp.gate.e_score_correction", # deepseek-v3
        ),

        # Feed-forward up
        MODEL_TENSOR.FFN_UP: (
            "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",                # gptneox
            "transformer.h.{bid}.mlp.c_fc",                           # gpt2 jais
            "transformer.blocks.{bid}.ffn.up_proj",                   # mpt
            "transformer.h.{bid}.mlp.dense_h_to_4h",                  # falcon
            "h.{bid}.mlp.dense_h_to_4h",                              # bloom
            "model.layers.{bid}.mlp.up_proj",                         # llama-hf refact nemotron olmo2
            "layers.{bid}.feed_forward.w3",                           # llama-pth
            "encoder.layer.{bid}.intermediate.dense",                 # bert
            "transformer.h.{bid}.mlp.fc_in",                          # gpt-j
            "transformer.h.{bid}.mlp.linear_3",                       # refact
            "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h",  # persimmon
            "model.layers.{bid}.mlp.dense_h_to_4h",                   # persimmon
            "transformer.h.{bid}.mlp.w1",                             # qwen
            "h.{bid}.mlp.c_fc",                                       # gpt2
            "transformer.h.{bid}.mlp.fc1",                            # phi2
            "model.layers.{bid}.mlp.fc1",                             # phi2
            "model.layers.{bid}.mlp.gate_up_proj",                    # phi3
            "model.layers.layers.{bid}.mlp.up_proj",                  # plamo
            "model.layers.{bid}.feed_forward.w3",                     # internlm2
            "encoder.layers.{bid}.mlp.fc11",                          # nomic-bert
            "model.layers.{bid}.mlp.c_fc",                            # starcoder2
            "encoder.layer.{bid}.mlp.gated_layers_v",                 # jina-bert-v2
            "model.layers.{bid}.residual_mlp.w3",                     # arctic
            "encoder.layers.{bid}.mlp.dense_h_to_4h",                 # chatglm
            "transformer.h.{bid}.mlp.c_fc_1",                         # exaone
        ),

        MODEL_TENSOR.FFN_UP_EXP: (
            "layers.{bid}.feed_forward.experts.w3",           # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear_v",   # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.v1",    # dbrx
            "model.layers.{bid}.mlp.experts.up_proj",         # qwen2moe olmoe (merged)
            "model.layers.{bid}.block_sparse_moe.experts.w3", # phimoe (merged)
        ),

        MODEL_TENSOR.FFN_UP_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert.up_proj",  # qwen2moe
            "model.layers.{bid}.mlp.shared_experts.up_proj", # deepseek deepseek2
        ),

        # AWQ-activation gate
        MODEL_TENSOR.FFN_ACT: (
            "transformer.blocks.{bid}.ffn.act",  # mpt
        ),

        # Feed-forward gate
        MODEL_TENSOR.FFN_GATE: (
            "model.layers.{bid}.mlp.gate_proj",           # llama-hf refact olmo2
            "layers.{bid}.feed_forward.w1",               # llama-pth
            "transformer.h.{bid}.mlp.w2",                 # qwen
            "transformer.h.{bid}.mlp.c_fc2",              # jais
            "model.layers.layers.{bid}.mlp.gate_proj",    # plamo
            "model.layers.{bid}.feed_forward.w1",         # internlm2
            "encoder.layers.{bid}.mlp.fc12",              # nomic-bert
            "encoder.layer.{bid}.mlp.gated_layers_w",     # jina-bert-v2
            "transformer.h.{bid}.mlp.linear_1",           # refact
            "model.layers.{bid}.residual_mlp.w1",         # arctic
            "transformer.h.{bid}.mlp.c_fc_0",             # exaone
        ),

        MODEL_TENSOR.FFN_GATE_EXP: (
            "layers.{bid}.feed_forward.experts.w1",           # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear",     # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.w1",    # dbrx
            "model.layers.{bid}.mlp.experts.gate_proj",       # qwen2moe olmoe (merged)
            "model.layers.{bid}.block_sparse_moe.experts.w1", # phimoe (merged)
        ),

        MODEL_TENSOR.FFN_GATE_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert.gate_proj",  # qwen2moe
            "model.layers.{bid}.mlp.shared_experts.gate_proj", # deepseek deepseek2
        ),

        # Feed-forward down
        MODEL_TENSOR.FFN_DOWN: (
            "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",                # gptneox
            "transformer.h.{bid}.mlp.c_proj",                         # gpt2 refact qwen jais
            "transformer.blocks.{bid}.ffn.down_proj",                 # mpt
            "transformer.h.{bid}.mlp.dense_4h_to_h",                  # falcon
            "h.{bid}.mlp.dense_4h_to_h",                              # bloom
            "model.layers.{bid}.mlp.down_proj",                       # llama-hf nemotron olmo2
            "layers.{bid}.feed_forward.w2",                           # llama-pth
            "encoder.layer.{bid}.output.dense",                       # bert
            "transformer.h.{bid}.mlp.fc_out",                         # gpt-j
            "language_model.encoder.layers.{bid}.mlp.dense_4h_to_h",  # persimmon
            "model.layers.{bid}.mlp.dense_4h_to_h",                   # persimmon
            "h.{bid}.mlp.c_proj",                                     # gpt2
            "transformer.h.{bid}.mlp.fc2",                            # phi2
            "model.layers.{bid}.mlp.fc2",                             # phi2
            "model.layers.layers.{bid}.mlp.down_proj",                # plamo
            "model.layers.{bid}.feed_forward.w2",                     # internlm2
            "encoder.layers.{bid}.mlp.fc2",                           # nomic-bert
            "model.layers.{bid}.mlp.c_proj",                          # starcoder2
            "encoder.layer.{bid}.mlp.wo",                             # jina-bert-v2
            "transformer.layers.{bid}.ffn.proj_2",                    # openelm
            "model.layers.{bid}.residual_mlp.w2",                     # arctic
            "encoder.layer.{bid}.mlp.down_layer",                     # jina-bert-v2
            "encoder.layers.{bid}.mlp.dense_4h_to_h",                 # chatglm
            "model.layers.h.{bid}.mlp.c_proj",                        # exaone
        ),

        MODEL_TENSOR.FFN_DOWN_EXP: (
            "layers.{bid}.feed_forward.experts.w2",              # mixtral (merged)
            "transformer.decoder_layer.{bid}.moe.linear_1",      # Grok (merged)
            "transformer.blocks.{bid}.ffn.experts.mlp.w2",       # dbrx
            "model.layers.{bid}.mlp.experts.down_proj",          # qwen2moe olmoe (merged)
            "model.layers.{bid}.block_sparse_moe.output_linear", # granitemoe
            "model.layers.{bid}.block_sparse_moe.experts.w2",    # phimoe (merged)
        ),

        MODEL_TENSOR.FFN_DOWN_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert.down_proj",  # qwen2moe
            "model.layers.{bid}.mlp.shared_experts.down_proj", # deepseek deepseek2
        ),

        MODEL_TENSOR.ATTN_Q_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.q_layernorm",
            "model.layers.{bid}.self_attn.q_layernorm",                       # persimmon
            "model.layers.{bid}.self_attn.q_norm",                            # cohere olmoe chameleon olmo2
            "transformer.blocks.{bid}.attn.q_ln",                             # sea-lion
            "encoder.layer.{bid}.attention.self.layer_norm_q",                # jina-bert-v2
            "transformer.layers.{bid}.attn.q_norm",                           # openelm
        ),

        MODEL_TENSOR.ATTN_K_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.k_layernorm",
            "model.layers.{bid}.self_attn.k_layernorm",                       # persimmon
            "model.layers.{bid}.self_attn.k_norm",                            # cohere olmoe chameleon olmo2
            "transformer.blocks.{bid}.attn.k_ln",                             # sea-lion
            "encoder.layer.{bid}.attention.self.layer_norm_k",                # jina-bert-v2
            "transformer.layers.{bid}.attn.k_norm",                           # openelm
        ),

        MODEL_TENSOR.ROPE_FREQS: (
            "language_model.encoder.layers.{bid}.self_attention.rotary_emb.inv_freq",  # persimmon
        ),

        MODEL_TENSOR.LAYER_OUT_NORM: (
            "encoder.layer.{bid}.output.LayerNorm",         # bert
            "encoder.layers.{bid}.norm2",                   # nomic-bert
            "transformer.decoder_layer.{bid}.rms_norm_3",   # Grok
            "encoder.layer.{bid}.mlp.layernorm",            # jina-bert-v2
            "encoder.layer.{bid}.layer_norm_2"              # jina-v2-code
        ),

        MODEL_TENSOR.SSM_IN: (
            "model.layers.{bid}.in_proj",
            "backbone.layers.{bid}.mixer.in_proj",
        ),

        MODEL_TENSOR.SSM_CONV1D: (
            "model.layers.{bid}.conv1d",
            "backbone.layers.{bid}.mixer.conv1d",
        ),

        MODEL_TENSOR.SSM_X: (
            "model.layers.{bid}.x_proj",
            "backbone.layers.{bid}.mixer.x_proj",
        ),

        MODEL_TENSOR.SSM_DT: (
            "model.layers.{bid}.dt_proj",
            "backbone.layers.{bid}.mixer.dt_proj",
        ),

        MODEL_TENSOR.SSM_A: (
            "model.layers.{bid}.A_log",
            "backbone.layers.{bid}.mixer.A_log",
        ),

        MODEL_TENSOR.SSM_D: (
            "model.layers.{bid}.D",
            "backbone.layers.{bid}.mixer.D",
        ),

        MODEL_TENSOR.SSM_OUT: (
            "model.layers.{bid}.out_proj",
            "backbone.layers.{bid}.mixer.out_proj",
        ),

        MODEL_TENSOR.TIME_MIX_W1: (
            "rwkv.blocks.{bid}.attention.time_maa_w1",  # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_w1", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_W2: (
            "rwkv.blocks.{bid}.attention.time_maa_w2",  # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_w2", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LERP_X: (
            "rwkv.blocks.{bid}.attention.time_maa_x",   # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_x",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LERP_K: (
            "rwkv.blocks.{bid}.attention.time_maa_k",   # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_k",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LERP_V: (
            "rwkv.blocks.{bid}.attention.time_maa_v",   # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_v",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LERP_R: (
            "rwkv.blocks.{bid}.attention.time_maa_r",   # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_r",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LERP_G: (
            "rwkv.blocks.{bid}.attention.time_maa_g",   # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_g",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LERP_W: (
            "rwkv.blocks.{bid}.attention.time_maa_w",   # rwkv v6
            "model.layers.{bid}.self_attn.time_maa_w",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_FIRST: (
            "rwkv.blocks.{bid}.attention.time_faaaa",   # rwkv v6
        ),

        MODEL_TENSOR.TIME_MIX_DECAY: (
            "rwkv.blocks.{bid}.attention.time_decay",   # rwkv v6
            "model.layers.{bid}.self_attn.time_decay",  # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_DECAY_W1: (
            "rwkv.blocks.{bid}.attention.time_decay_w1",  # rwkv v6
            "model.layers.{bid}.self_attn.time_decay_w1", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_DECAY_W2: (
            "rwkv.blocks.{bid}.attention.time_decay_w2",  # rwkv v6
            "model.layers.{bid}.self_attn.time_decay_w2", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_KEY: (
            "rwkv.blocks.{bid}.attention.key",     # rwkv
            "model.layers.{bid}.self_attn.k_proj", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_VALUE: (
            "rwkv.blocks.{bid}.attention.value",   # rwkv
            "model.layers.{bid}.self_attn.v_proj", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_RECEPTANCE: (
            "rwkv.blocks.{bid}.attention.receptance", # rwkv
            "model.layers.{bid}.self_attn.q_proj",    # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_GATE: (
            "rwkv.blocks.{bid}.attention.gate",  # rwkv
            "model.layers.{bid}.self_attn.gate", # rwkv6qwen2
        ),

        MODEL_TENSOR.TIME_MIX_LN: (
            "rwkv.blocks.{bid}.attention.ln_x", # rwkv
        ),

        MODEL_TENSOR.TIME_MIX_OUTPUT: (
            "rwkv.blocks.{bid}.attention.output",  # rwkv
            "model.layers.{bid}.self_attn.o_proj", # rwkv6qwen2
        ),

        MODEL_TENSOR.CHANNEL_MIX_LERP_K: (
            "rwkv.blocks.{bid}.feed_forward.time_maa_k", # rwkv v6
        ),

        MODEL_TENSOR.CHANNEL_MIX_LERP_R: (
            "rwkv.blocks.{bid}.feed_forward.time_maa_r", # rwkv v6
        ),

        MODEL_TENSOR.CHANNEL_MIX_KEY: (
            "rwkv.blocks.{bid}.feed_forward.key", # rwkv
        ),

        MODEL_TENSOR.CHANNEL_MIX_RECEPTANCE: (
            "rwkv.blocks.{bid}.feed_forward.receptance", # rwkv
        ),

        MODEL_TENSOR.CHANNEL_MIX_VALUE: (
            "rwkv.blocks.{bid}.feed_forward.value", # rwkv
        ),

        MODEL_TENSOR.ATTN_Q_A: (
            "model.layers.{bid}.self_attn.q_a_proj", # deepseek2
        ),

        MODEL_TENSOR.ATTN_Q_B: (
            "model.layers.{bid}.self_attn.q_b_proj", # deepseek2
        ),

        MODEL_TENSOR.ATTN_KV_A_MQA: (
            "model.layers.{bid}.self_attn.kv_a_proj_with_mqa", # deepseek2
        ),

        MODEL_TENSOR.ATTN_KV_B: (
            "model.layers.{bid}.self_attn.kv_b_proj", # deepseek2
        ),

        MODEL_TENSOR.ATTN_Q_A_NORM: (
            "model.layers.{bid}.self_attn.q_a_layernorm", # deepseek2
        ),

        MODEL_TENSOR.ATTN_KV_A_NORM: (
            "model.layers.{bid}.self_attn.kv_a_layernorm", # deepseek2
        ),

        MODEL_TENSOR.ATTN_SUB_NORM: (
            "model.layers.{bid}.self_attn.inner_attn_ln",  # bitnet
        ),

        MODEL_TENSOR.FFN_SUB_NORM: (
            "model.layers.{bid}.mlp.ffn_layernorm",  # bitnet
        ),

        MODEL_TENSOR.DEC_ATTN_NORM: (
            "decoder.block.{bid}.layer.0.layer_norm", # t5
        ),

        MODEL_TENSOR.DEC_ATTN_Q: (
            "decoder.block.{bid}.layer.0.SelfAttention.q", # t5
        ),

        MODEL_TENSOR.DEC_ATTN_K: (
            "decoder.block.{bid}.layer.0.SelfAttention.k", # t5
        ),

        MODEL_TENSOR.DEC_ATTN_V: (
            "decoder.block.{bid}.layer.0.SelfAttention.v", # t5
        ),

        MODEL_TENSOR.DEC_ATTN_OUT: (
            "decoder.block.{bid}.layer.0.SelfAttention.o", # t5
        ),

        MODEL_TENSOR.DEC_ATTN_REL_B: (
            "decoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias", # t5
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_NORM: (
            "decoder.block.{bid}.layer.1.layer_norm", # t5
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_Q: (
            "decoder.block.{bid}.layer.1.EncDecAttention.q", # t5
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_K: (
            "decoder.block.{bid}.layer.1.EncDecAttention.k", # t5
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_V: (
            "decoder.block.{bid}.layer.1.EncDecAttention.v", # t5
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_OUT: (
            "decoder.block.{bid}.layer.1.EncDecAttention.o", # t5
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_REL_B: (
            "decoder.block.{bid}.layer.1.EncDecAttention.relative_attention_bias", # t5
        ),

        MODEL_TENSOR.DEC_FFN_NORM: (
            "decoder.block.{bid}.layer.2.layer_norm", # t5
        ),

        MODEL_TENSOR.DEC_FFN_GATE: (
            "decoder.block.{bid}.layer.2.DenseReluDense.wi_0", # flan-t5
        ),

        MODEL_TENSOR.DEC_FFN_UP: (
            "decoder.block.{bid}.layer.2.DenseReluDense.wi",   # t5
            "decoder.block.{bid}.layer.2.DenseReluDense.wi_1", # flan-t5
        ),

        MODEL_TENSOR.DEC_FFN_DOWN: (
            "decoder.block.{bid}.layer.2.DenseReluDense.wo", # t5
        ),

        MODEL_TENSOR.DEC_OUTPUT_NORM: (
            "decoder.final_layer_norm", # t5
        ),

        MODEL_TENSOR.ENC_ATTN_NORM: (
            "encoder.block.{bid}.layer.0.layer_norm", # t5
        ),

        MODEL_TENSOR.ENC_ATTN_Q: (
            "encoder.block.{bid}.layer.0.SelfAttention.q", # t5
        ),

        MODEL_TENSOR.ENC_ATTN_K: (
            "encoder.block.{bid}.layer.0.SelfAttention.k", # t5
        ),

        MODEL_TENSOR.ENC_ATTN_V: (
            "encoder.block.{bid}.layer.0.SelfAttention.v", # t5
        ),

        MODEL_TENSOR.ENC_ATTN_OUT: (
            "encoder.block.{bid}.layer.0.SelfAttention.o", # t5
        ),

        MODEL_TENSOR.ENC_ATTN_REL_B: (
            "encoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias", # t5
        ),

        MODEL_TENSOR.ENC_FFN_NORM: (
            "encoder.block.{bid}.layer.1.layer_norm", # t5
        ),

        MODEL_TENSOR.ENC_FFN_GATE: (
            "encoder.block.{bid}.layer.1.DenseReluDense.wi_0", # flan-t5
        ),

        MODEL_TENSOR.ENC_FFN_UP: (
            "encoder.block.{bid}.layer.1.DenseReluDense.wi",   # t5
            "encoder.block.{bid}.layer.1.DenseReluDense.wi_1", # flan-t5
        ),

        MODEL_TENSOR.ENC_FFN_DOWN: (
            "encoder.block.{bid}.layer.1.DenseReluDense.wo", # t5
        ),

        ############################################################################
        # TODO: these do not belong to block_mappings_cfg - move them to mappings_cfg
        MODEL_TENSOR.ENC_OUTPUT_NORM: (
            "encoder.final_layer_norm", # t5
        ),

        MODEL_TENSOR.CLS: (
            "classifier",       # jina
            "classifier.dense", # roberta
        ),

        MODEL_TENSOR.CLS_OUT: (
            "classifier.out_proj", # roberta
        ),
        #############################################################################

        MODEL_TENSOR.CONVNEXT_DW: (
            "backbone.convnext.{bid}.dwconv", # wavtokenizer
        ),

        MODEL_TENSOR.CONVNEXT_NORM: (
            "backbone.convnext.{bid}.norm", # wavtokenizer
        ),

        MODEL_TENSOR.CONVNEXT_PW1: (
            "backbone.convnext.{bid}.pwconv1", # wavtokenizer
        ),

        MODEL_TENSOR.CONVNEXT_PW2: (
            "backbone.convnext.{bid}.pwconv2", # wavtokenizer
        ),

        MODEL_TENSOR.CONVNEXT_GAMMA: (
            "backbone.convnext.{bid}.gamma", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_CONV1: (
            "backbone.posnet.{bid}.conv1", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_CONV2: (
            "backbone.posnet.{bid}.conv2", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_NORM: (
            "backbone.posnet.{bid}.norm", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_NORM1: (
            "backbone.posnet.{bid}.norm1", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_NORM2: (
            "backbone.posnet.{bid}.norm2", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_ATTN_NORM: (
            "backbone.posnet.{bid}.norm", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_ATTN_Q: (
            "backbone.posnet.{bid}.q", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_ATTN_K: (
            "backbone.posnet.{bid}.k", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_ATTN_V: (
            "backbone.posnet.{bid}.v", # wavtokenizer
        ),

        MODEL_TENSOR.POSNET_ATTN_OUT: (
            "backbone.posnet.{bid}.proj_out", # wavtokenizer
        ),

        #############################################################################

        MODEL_TENSOR.V_MMPROJ: (
            "multi_modal_projector.linear_{bid}",
        ),

        MODEL_TENSOR.V_MMPROJ_FC: (
            "model.connector.modality_projection.proj", # SmolVLM
        ),

        MODEL_TENSOR.V_MMPROJ_MLP: (
            "model.mm_projector.mlp.mlp.{bid}",
        ),

        MODEL_TENSOR.V_MMPROJ_PEG: (
            "model.mm_projector.peg.peg.{bid}",
        ),

        MODEL_TENSOR.V_ENC_EMBD_CLS: (
            "vision_tower.vision_model.embeddings.class_embedding",
        ),

        MODEL_TENSOR.V_ENC_EMBD_PATCH: (
            "vision_tower.vision_model.embeddings.patch_embedding",
            "vpm.embeddings.patch_embedding",
            "model.vision_model.embeddings.patch_embedding", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_EMBD_POS: (
            "vision_tower.vision_model.embeddings.position_embedding",
            "vpm.embeddings.position_embedding",
            "model.vision_model.embeddings.position_embedding", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_ATTN_Q: (
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.q_proj",
            "vpm.encoder.layers.{bid}.self_attn.q_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.q_proj", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_ATTN_K: (
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.k_proj",
            "vpm.encoder.layers.{bid}.self_attn.k_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.k_proj", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_ATTN_V: (
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.v_proj",
            "vpm.encoder.layers.{bid}.self_attn.v_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.v_proj", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_INPUT_NORM: (
            "vision_tower.vision_model.encoder.layers.{bid}.layer_norm1",
            "vpm.encoder.layers.{bid}.layer_norm1",
            "model.vision_model.encoder.layers.{bid}.layer_norm1", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_OUTPUT: (
            "vision_tower.vision_model.encoder.layers.{bid}.self_attn.out_proj",
            "vpm.encoder.layers.{bid}.self_attn.out_proj",
            "model.vision_model.encoder.layers.{bid}.self_attn.out_proj", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_OUTPUT_NORM: (
            "vision_tower.vision_model.encoder.layers.{bid}.layer_norm2",
            "vpm.encoder.layers.{bid}.layer_norm2",
            "model.vision_model.encoder.layers.{bid}.layer_norm2", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_FFN_UP: (
            "vision_tower.vision_model.encoder.layers.{bid}.mlp.fc1",
            "vpm.encoder.layers.{bid}.mlp.fc1",
            "model.vision_model.encoder.layers.{bid}.mlp.fc1", # SmolVLM
        ),

        MODEL_TENSOR.V_ENC_FFN_DOWN: (
            "vision_tower.vision_model.encoder.layers.{bid}.mlp.fc2",
            "vpm.encoder.layers.{bid}.mlp.fc2",
            "model.vision_model.encoder.layers.{bid}.mlp.fc2", # SmolVLM
        ),

        MODEL_TENSOR.V_PRE_NORM: (
            "vision_tower.vision_model.pre_layrnorm",
        ),

        MODEL_TENSOR.V_POST_NORM: (
            "vision_tower.vision_model.post_layernorm",
            "model.vision_model.post_layernorm", # SmolVLM
        ),

        MODEL_TENSOR.V_RESMPL_POS_EMBD_K: (
            "resampler.pos_embed_k",
        ),

        MODEL_TENSOR.V_RESMPL_ATTN_Q: (
            "resampler.attn.in_proj_q", # tensor generated from resampler.attn.in_proj
        ),

        MODEL_TENSOR.V_RESMPL_ATTN_K: (
            "resampler.attn.in_proj_k", # tensor generated from resampler.attn.in_proj
        ),

        MODEL_TENSOR.V_RESMPL_ATTN_V: (
            "resampler.attn.in_proj_v", # tensor generated from resampler.attn.in_proj
        ),

        MODEL_TENSOR.V_RESMPL_ATTN_OUT: (
            "resampler.attn.out_proj",
        ),

        MODEL_TENSOR.V_RESMPL_KV: (
            "resampler.kv_proj",
        ),

        MODEL_TENSOR.V_RESMPL_POST_NORM: (
            "resampler.ln_post",
        ),

        MODEL_TENSOR.V_RESMPL_KV_NORM: (
            "resampler.ln_kv",
        ),

        MODEL_TENSOR.V_RESMPL_Q_NORM: (
            "resampler.ln_q",
        ),

        MODEL_TENSOR.V_RESMPL_PROJ: (
            "resampler.proj",
        ),

        MODEL_TENSOR.V_RESMPL_QUERY: (
            "resampler.query",
        ),

        MODEL_TENSOR.V_TOK_EMBD_IMAGE:(
            "v.tok_embd.image", # tensor generated from token embeddings
        ),

        MODEL_TENSOR.V_TOK_EMBD_END_IMAGE:(
            "v.tok_embd.end_image", # tensor generated from token embeddings
        ),

        MODEL_TENSOR.V_TOK_EMBD_SLICE:(
            "v.tok_embd.slice", # tensor generated from token embeddings
        ),

        MODEL_TENSOR.V_TOK_EMBD_END_SLICE:(
            "v.tok_embd.end_slice", # tensor generated from token embeddings
        ),
    }

    # architecture-specific block mappings
    arch_block_mappings_cfg: dict[MODEL_ARCH, dict[MODEL_TENSOR, tuple[str, ...]]] = {
        MODEL_ARCH.ARCTIC: {
            MODEL_TENSOR.FFN_NORM: (
                "model.layers.{bid}.residual_layernorm",
            ),
            MODEL_TENSOR.FFN_NORM_EXP: (
                "model.layers.{bid}.post_attention_layernorm",
            ),
        },
    }

    mapping: dict[str, tuple[MODEL_TENSOR, str]]

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
        if arch in self.arch_block_mappings_cfg:
            self.block_mappings_cfg.update(self.arch_block_mappings_cfg[arch])
        for bid in range(n_blocks):
            for tensor, keys in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue

                tensor_name = TENSOR_NAMES[tensor].format(bid = bid)
                self.mapping[tensor_name] = (tensor, tensor_name)
                for key in keys:
                    key = key.format(bid = bid)
                    self.mapping[key] = (tensor, tensor_name)

    def get_type_and_name(self, key: str, try_suffixes: Sequence[str] = ()) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[:-len(suffix)])
                if result is not None:
                    return result[0], result[1] + suffix
        return None

    def get_name(self, key: str, try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type(self, key: str, try_suffixes: Sequence[str] = ()) -> MODEL_TENSOR | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[0]

    def __getitem__(self, key: str) -> str:
        try:
            return self.mapping[key][1]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return repr(self.mapping)


def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)
