# Recommended mapping of model tensor names for storage in gguf

def get_tensor_map( n_blocks : int):
    tensor_map = {}
    # Token embeddings
    mapped_to = "transformer.token_embd"
    tensor_map["gpt_neox.embed_in"] = mapped_to           # gptneox
    tensor_map["transformer.wte"] = mapped_to             # gpt2 mpt
    tensor_map["transformer.word_embeddings"] = mapped_to # falcon
    tensor_map["model.embed_tokens"] = mapped_to          # llama-hf
    tensor_map["tok_embeddings"] = mapped_to              # llama-pth
    # Position embeddings
    mapped_to = "transformer.pos_embd"
    tensor_map["transformer.wpe"] = mapped_to # gpt2
    # Output norm
    mapped_to = "transformer.output_norm"
    tensor_map["gpt_neox.final_layer_norm"] = mapped_to # gptneox
    tensor_map["transformer.ln_f"] = mapped_to          # gpt2 falcon
    tensor_map["transformer.norm_f"] = mapped_to        # mpt
    tensor_map["model.norm"] = mapped_to                # llama-hf
    tensor_map["norm"] = mapped_to                      # llama-pth
    # Output
    mapped_to = "transformer.output"
    tensor_map["embed_out"] = mapped_to # gptneox
    tensor_map["lm_head"] = mapped_to   # gpt2 mpt falcon llama-hf
    tensor_map["output"] = mapped_to    # llama-pth
    # Attention and fee-forward layer blocks
    for i in range(0,n_blocks):
        # Attention norm
        mapped_to = "transformer.blocks."+str(i)+".attn_norm"
        tensor_map["gpt_neox.layers."+str(i)+".input_layernorm"] = mapped_to # gptneox
        tensor_map["transformer.h."+str(i)+".ln_1"] = mapped_to              # gpt2
        tensor_map["transformer.blocks."+str(i)+".norm_1"] = mapped_to       # mpt
        tensor_map["transformer.h."+str(i)+".input_layernorm"] = mapped_to   # falcon7b
        tensor_map["transformer.h."+str(i)+".ln_attn"] = mapped_to           # falcon40b
        tensor_map["model.layers."+str(i)+".input_layernorm"] = mapped_to    # llama-hf
        tensor_map["layers."+str(i)+".attention_norm"] = mapped_to           # llama-pth
        # Attention norm 2
        mapped_to = "transformer.blocks."+str(i)+".attn_norm_2"
        tensor_map["transformer.h."+str(i)+".ln_mlp"] = mapped_to # falcon40b
        # Attention query-key-value
        mapped_to = "transformer.blocks."+str(i)+".attn_qkv"
        tensor_map["gpt_neox.layers."+str(i)+".attention.query_key_value"] = mapped_to     # gptneox
        tensor_map["transformer.h."+str(i)+".attn.c_attn"] = mapped_to                     # gpt2
        tensor_map["transformer.blocks."+str(i)+".attn.Wqkv"] = mapped_to                  # mpt
        tensor_map["transformer.h."+str(i)+".self_attention.query_key_value"] = mapped_to  # falcon
        # Attention query
        mapped_to = "transformer.blocks."+str(i)+".attn_q"
        tensor_map["model.layers."+str(i)+".self_attn.q_proj"] = mapped_to # llama-hf
        tensor_map["layers."+str(i)+".attention.wq"] = mapped_to           # llama-pth
        # Attention key
        mapped_to = "transformer.blocks."+str(i)+".attn_k"
        tensor_map["model.layers."+str(i)+".self_attn.k_proj"] = mapped_to # llama-hf
        tensor_map["layers."+str(i)+".attention.wk"] = mapped_to           # llama-pth
        # Attention value
        mapped_to = "transformer.blocks."+str(i)+".attn_v"
        tensor_map["model.layers."+str(i)+".self_attn.v_proj"] = mapped_to # llama-hf
        tensor_map["layers."+str(i)+".attention.wv"] = mapped_to           # llama-pth
        # Attention output
        mapped_to = "transformer.blocks."+str(i)+".attn_output"
        tensor_map["gpt_neox.layers."+str(i)+".attention.dense"] = mapped_to    # gptneox
        tensor_map["transformer.h."+str(i)+".attn.c_proj"] = mapped_to          # gpt2
        tensor_map["transformer.blocks."+str(i)+".attn.out_proj"] = mapped_to   # mpt
        tensor_map["transformer.h."+str(i)+".self_attention.dense"] = mapped_to # falcon
        tensor_map["model.layers."+str(i)+".self_attn.o_proj"] = mapped_to      # llama-hf
        tensor_map["layers."+str(i)+".attention.wo"] = mapped_to                # llama-pth
        # Feed-forward norm
        mapped_to = "transformer.blocks."+str(i)+".ffn_norm"
        tensor_map["gpt_neox.layers."+str(i)+".post_attention_layernorm"] = mapped_to # gptneox
        tensor_map["transformer.h."+str(i)+".ln_2"] = mapped_to                       # gpt2
        tensor_map["transformer.blocks."+str(i)+".norm_2"] = mapped_to               # mpt
        tensor_map["model.layers."+str(i)+".post_attention_layernorm"] = mapped_to    # llama-hf
        tensor_map["layers."+str(i)+".ffn_norm"] = mapped_to                          # llama-pth
        # Feed-forward up
        mapped_to = "transformer.blocks."+str(i)+".ffn_up"
        tensor_map["gpt_neox.layers."+str(i)+".mlp.dense_h_to_4h"] = mapped_to # gptneox
        tensor_map["transformer.h."+str(i)+".mlp.c_fc"] = mapped_to            # gpt2
        tensor_map["transformer.blocks."+str(i)+".ffn.up_proj"] = mapped_to    # mpt
        tensor_map["transformer.h."+str(i)+".mlp.dense_h_to_4h"] = mapped_to   # falcon
        tensor_map["model.layers."+str(i)+".mlp.up_proj"] = mapped_to          # llama-hf
        tensor_map["layers."+str(i)+".feed_forward.w3"] = mapped_to            # llama-pth
        # Feed-forward gate
        mapped_to = "transformer.blocks."+str(i)+".ffn_gate"
        tensor_map["model.layers."+str(i)+".mlp.gate_proj"] = mapped_to # llama-hf
        tensor_map["layers."+str(i)+".feed_forward.w1"] = mapped_to     # llama-pth
        # Feed-forward down
        mapped_to = "transformer.blocks."+str(i)+".ffn_down"
        tensor_map["gpt_neox.layers."+str(i)+".mlp.dense_4h_to_h"] = mapped_to # gptneox
        tensor_map["transformer.h."+str(i)+".mlp.c_proj"] = mapped_to          # gpt2
        tensor_map["transformer.blocks."+str(i)+".ffn.down_proj"] = mapped_to  # mpt
        tensor_map["transformer.h."+str(i)+".mlp.dense_4h_to_h"] = mapped_to   # falcon
        tensor_map["model.layers."+str(i)+".mlp.down_proj"] = mapped_to        # llama-hf
        tensor_map["layers."+str(i)+".feed_forward.w2"] = mapped_to            # llama-pth

    return tensor_map

