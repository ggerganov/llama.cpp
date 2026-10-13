#!/bin/bash


run() { fails=0; for i in $(seq 50); do
  env "$@" GGML_SCHED_DEBUG_REALLOC=1 build-release/bin/test-recurrent-state-rollback \
    -m build-release/tests/test-models/lfm2moe-moe.gguf >/dev/null 2>&1 || fails=$((fails+1)); done
  echo "$* -> $fails/50"; }

#run X=baseline
#run GGML_VK_DISABLE_FUSION=1          # 0 になれば topk_moe などの fusion の問題
#run GGML_VK_MAX_NODES_PER_SUBMIT=1    # 結果が変われば submit 単位やタイミングに依存
#run GGML_VK_SERIALIZE_SUBMISSIONS=1
#run GGML_VK_SERIALIZE_SUBMISSIONS=1 GGML_VK_DISABLE_FUSION=1
#run GGML_VK_DISABLE_FUSION=1 GGML_VK_FORCE_SYNC=1
#run GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_ASYNC=1
#run GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_DESCRIPTOR_REUSE=1
#run GGML_VK_DISABLE_FUSION=1 GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
#run GGML_VK_DISABLE_FUSION=1 GGML_VK_NO_HOST_ASYNC_CPY=1
run GGML_VK_DISABLE_FUSION=1 GGML_VK_SYNC_GRAPH_END=1
run GGML_VK_DISABLE_FUSION=1 GGML_VK_NO_GET_ASYNC=1
