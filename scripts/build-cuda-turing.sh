#!/usr/bin/env bash
#
# Build llama-server (and llama-cli) with the CUDA + CPU backends for an NVIDIA Turing GPU
# (compute capability 7.5: RTX 20xx, GTX 16xx, Tesla T4, Quadro RTX), with the MMQ kernels forced.
#
#   ./build-cuda-turing.sh [extra cmake args...]
#
# Requirements: CUDA toolkit 13.x (nvcc), a host compiler that nvcc accepts (gcc 16 here), cmake, ninja.
# Nothing is downloaded, the sources of this repository are used as they are.

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build-cuda-turing}"
JOBS="${JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------

# the host compiler nvcc drives, override with: CUDA_HOST_COMPILER=/usr/bin/g++-15 ./build-cuda-turing.sh
CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER:-}"
if [[ -z "${CUDA_HOST_COMPILER}" ]]; then
    for c in /usr/bin/g++-16 /usr/bin/g++-15 /usr/bin/g++-14 g++-16 g++; do
        if command -v "$c" > /dev/null 2>&1; then CUDA_HOST_COMPILER="$(command -v "$c")"; break; fi
    done
fi

# nvcc, override with: CUDACXX=/usr/local/cuda-13.4/bin/nvcc ./build-cuda-turing.sh
CUDACXX="${CUDACXX:-$(command -v nvcc || true)}"
if [[ -z "${CUDACXX}" ]]; then
    echo "error: nvcc not found, set CUDACXX=/path/to/nvcc" >&2
    exit 1
fi

echo "== host compiler : ${CUDA_HOST_COMPILER} ($(${CUDA_HOST_COMPILER} -dumpversion 2>/dev/null || echo '?'))"
echo "== nvcc          : ${CUDACXX}"
"${CUDACXX}" --version | tail -2
echo "== gpu           : $(command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1 || echo 'nvidia-smi not found')"
echo "== build dir     : ${BUILD_DIR}"
echo

# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------
#
# CMAKE_CUDA_ARCHITECTURES=75-real  -> only sm_75 SASS, no PTX: the smallest and fastest build for
#                                      a single Turing GPU. Use "75" to also embed PTX for future
#                                      drivers, or "native" to let nvcc detect the GPU.
# GGML_CUDA_FORCE_MMQ=ON            -> the quantized matmuls use the MMQ kernels instead of cuBLAS:
#                                      less VRAM, usually faster for the small batches of the decode.
# GGML_CUDA_F16=ON                  -> also compile the F16 matmul kernels (bigger build, more speed).
# GGML_NATIVE=OFF                   -> do not bake -march=native in, so the build stays portable.
# LLAMA_CURL=OFF                    -> no libcurl dependency for the model download.

cmake -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75-real \
    -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DCUDA_HOST_COMPILER="${CUDA_HOST_COMPILER}" \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    "$@"

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

ninja -C "${BUILD_DIR}" -j "${JOBS}" llama-server llama-cli llama-perplexity test-backend-ops test-moe-cache-prior

echo
echo "== done: $(pwd)/${BUILD_DIR}/bin"
ls -l "${BUILD_DIR}/bin" | sed -n '1,20p'

cat <<EOF

== verify that it really is a CUDA + CPU build ==============================
  # 1. the binary links the CUDA runtime and the ggml-cuda backend
  ldd build-cuda-turing/bin/llama-server | grep -Ei 'cuda|cublas|ggml'

  # 2. the CUDA kernels were compiled for sm_75 only
  cuobjdump --list-elf build-cuda-turing/bin/llama-server | head
  cuobjdump --list-ptx build-cuda-turing/bin/llama-server | head   # empty with 75-real

  # 3. the backends that the binary sees at runtime
  ./${BUILD_DIR}/bin/llama-server --list-devices

  # 4. the ops of the cache prior run on the GPU
  ./${BUILD_DIR}/bin/test-backend-ops test -b CUDA0 -o REPEAT,MUL,ADD,SUB,SCALE,CLAMP,SOFT_MAX,SUM_ROWS,ARGSORT,GET_ROWS,SET_ROWS,FILL,CONT,RESHAPE

  # 5. the cache-aware routing itself
  ./${BUILD_DIR}/bin/test-moe-cache-prior
=============================================================================
EOF
