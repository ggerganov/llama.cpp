Variable mapping from llama.c to ggml llama.cpp

config variables in llama.c
`dim`, `vocab_size`, `num_layers`, `num_heads`, `num_kv_heads`, `seq_length`

| llama.c (karpathy) | ggml (gg)     |dim|
| ------------- | ------------- |-- |
| `dim`  | `n_embed`  | Transformer dim |
| `hidden_dim`  | `n_ff` (calculated)  | ff hidden dim |
| `n_layers`  | `n_layers`  | number of decoder layers |
| `n_heads`  | `n_head`  | number of heads |
| `n_kv_heads`  | `-`  |  |
| `vocab_size`  | `n_vocab`  |  |
| `seq_len`  | `-`  |  |
| ---  | ---  | --- |
| `rms_att_weight`  | `attention_norm`  | `num_layers` x `dim`  |
| `rms_ffn_weight`  | `ffn_norm`  | `num_layers` x `dim`  |
| `wq`  | `ffn_norm`  |  `num_layers` x `dim` x `dim` |
| `qk`  | `ffn_norm`  |  `num_layers` x `dim` x `dim` |
| `wv`  | `ffn_norm`  |  `num_layers` x `dim` x `dim` |
| `wo`  | `wo`  |  `num_layers` x `dim` x `dim` |
| `w1`  | `w1`  |  `num_layers` x `hidden_dim` x `dim` |
| `w2`  | `w2`  |  `num_layers` x `dim` x `hidden_dim` |
| `w3`  | `w3`  |  `num_layers` x `hidden_dim` x `dim` |
| `token_embedding_table`  | `tok_embeddings`  | `vocab_size` x `dim`  |
| `rms_final_weight`  | `?`  |  `dim` |
| `freq_cis_real`  | `?`  | `seq_len` x `dim/2`  |
| `freq_cis_img `  | `?`  | `seq_len` x `dim/2`  |



