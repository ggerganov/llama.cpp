# Define the default target now so that it is always the first target
BUILD_TARGETS = main quantize quantize-stats perplexity embedding vdot train-text-from-scratch convert-llama2c-to-ggml simple server embd-input-test

# Binaries only useful for tests
TEST_TARGETS = tests/test-grammar-parser tests/test-double-float tests/test-grad0 tests/test-opt tests/test-quantize-fns tests/test-quantize-perf tests/test-sampling tests/test-tokenizer-0

default: $(BUILD_TARGETS)

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
#

# keep standard at C11 and C++11
# -Ofast tends to produce faster code, but may not be available for some compilers.
ifdef LLAMA_FAST
OPT = -Ofast
else
OPT = -O3
endif
CFLAGS   = -I.              $(OPT) -std=c11   -fPIC
CXXFLAGS = -I. -I./examples $(OPT) -std=c++11 -fPIC
LDFLAGS  =

ifdef LLAMA_DEBUG
	CFLAGS   += -O0 -g
	CXXFLAGS += -O0 -g
	LDFLAGS  += -g
else
	CFLAGS   += -DNDEBUG
	CXXFLAGS += -DNDEBUG
endif

ifdef LLAMA_SERVER_VERBOSE
	CXXFLAGS += -DSERVER_VERBOSE=$(LLAMA_SERVER_VERBOSE)
endif

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith \
			-Wmissing-prototypes
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# detect Windows
ifneq ($(findstring _NT,$(UNAME_S)),)
	_WIN32 := 1
endif

# library name prefix
ifneq ($(_WIN32),1)
	LIB_PRE := lib
endif

# Dynamic Shared Object extension
ifneq ($(_WIN32),1)
	DSO_EXT := .so
else
	DSO_EXT := .dll
endif

# Windows Sockets 2 (Winsock) for network-capable apps
ifeq ($(_WIN32),1)
	LWINSOCK2 := -lws2_32
endif

ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifdef LLAMA_PERF
	CFLAGS   += -DGGML_PERF
	CXXFLAGS += -DGGML_PERF
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686 amd64))
	# Use all CPU extensions that are available:
	CFLAGS   += -march=native -mtune=native
	CXXFLAGS += -march=native -mtune=native

	# Usage AVX-only
	#CFLAGS   += -mfma -mf16c -mavx
	#CXXFLAGS += -mfma -mf16c -mavx

	# Usage SSSE3-only (Not is SSE3!)
	#CFLAGS   += -mssse3
	#CXXFLAGS += -mssse3
endif

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	CFLAGS   += -mcpu=native
	CXXFLAGS += -mcpu=native
endif

ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, Zero
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 2
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif

ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 3, 4, Zero 2 (32-bit)
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS   += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif

ifndef LLAMA_NO_K_QUANTS
	CFLAGS   += -DGGML_USE_K_QUANTS
	CXXFLAGS += -DGGML_USE_K_QUANTS
	OBJS     += k_quants.o
ifdef LLAMA_QKK_64
	CFLAGS   += -DGGML_QKK_64
	CXXFLAGS += -DGGML_QKK_64
endif
endif

ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif # LLAMA_NO_ACCELERATE

ifdef LLAMA_MPI
	CFLAGS += -DGGML_USE_MPI -Wno-cast-qual
	CXXFLAGS += -DGGML_USE_MPI -Wno-cast-qual
	OBJS     += ggml-mpi.o
endif # LLAMA_MPI

ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS $(shell pkg-config --cflags openblas)
	LDFLAGS += $(shell pkg-config --libs openblas)
endif # LLAMA_OPENBLAS

ifdef LLAMA_BLIS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/blis -I/usr/include/blis
	LDFLAGS += -lblis -L/usr/local/lib
endif # LLAMA_BLIS

ifdef LLAMA_CUBLAS
	CFLAGS    += -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	CXXFLAGS  += -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	LDFLAGS   += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib
	OBJS      += ggml-cuda.o
	NVCCFLAGS = --forward-unknown-to-host-compiler -use_fast_math
ifdef LLAMA_CUDA_NVCC
	NVCC = $(LLAMA_CUDA_NVCC)
else
	NVCC = nvcc
endif #LLAMA_CUDA_NVCC
ifdef CUDA_DOCKER_ARCH
	NVCCFLAGS += -Wno-deprecated-gpu-targets -arch=$(CUDA_DOCKER_ARCH)
else
	NVCCFLAGS += -arch=native
endif # CUDA_DOCKER_ARCH
ifdef LLAMA_CUDA_FORCE_DMMV
	NVCCFLAGS += -DGGML_CUDA_FORCE_DMMV
endif # LLAMA_CUDA_FORCE_DMMV
ifdef LLAMA_CUDA_DMMV_X
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
else
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # LLAMA_CUDA_DMMV_X
ifdef LLAMA_CUDA_MMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_MMV_Y)
else ifdef LLAMA_CUDA_DMMV_Y
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=$(LLAMA_CUDA_DMMV_Y) # for backwards compatibility
else
	NVCCFLAGS += -DGGML_CUDA_MMV_Y=1
endif # LLAMA_CUDA_MMV_Y
ifdef LLAMA_CUDA_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_F16
ifdef LLAMA_CUDA_DMMV_F16
	NVCCFLAGS += -DGGML_CUDA_F16
endif # LLAMA_CUDA_DMMV_F16
ifdef LLAMA_CUDA_KQUANTS_ITER
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
else
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif
#ifdef LLAMA_CUDA_CUBLAS
#	NVCCFLAGS += -DGGML_CUDA_CUBLAS
#endif # LLAMA_CUDA_CUBLAS
ifdef LLAMA_CUDA_CCBIN
	NVCCFLAGS += -ccbin $(LLAMA_CUDA_CCBIN)
endif
ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
	$(NVCC) $(NVCCFLAGS) $(subst -Ofast,-O3,$(CXXFLAGS)) -Wno-pedantic -c $< -o $@
endif # LLAMA_CUBLAS

ifdef LLAMA_CLBLAST

	CFLAGS   += -DGGML_USE_CLBLAST $(shell pkg-config --cflags clblast OpenCL)
	CXXFLAGS += -DGGML_USE_CLBLAST $(shell pkg-config --cflags clblast OpenCL)

	# Mac provides OpenCL as a framework
	ifeq ($(UNAME_S),Darwin)
		LDFLAGS += -lclblast -framework OpenCL
	else
		LDFLAGS += $(shell pkg-config --libs clblast OpenCL)
	endif
	OBJS    += ggml-opencl.o

ggml-opencl.o: ggml-opencl.cpp ggml-opencl.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
endif # LLAMA_CLBLAST

ifdef LLAMA_METAL
	CFLAGS   += -DGGML_USE_METAL -DGGML_METAL_NDEBUG
	CXXFLAGS += -DGGML_USE_METAL
	LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
	OBJS     += ggml-metal.o
endif # LLAMA_METAL

ifdef LLAMA_METAL
ggml-metal.o: ggml-metal.m ggml-metal.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_METAL

ifdef LLAMA_MPI
ggml-mpi.o: ggml-mpi.c ggml-mpi.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_MPI

ifdef LLAMA_NO_K_QUANTS
k_quants.o: k_quants.c k_quants.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_NO_K_QUANTS

#
# Print build information
#

$(info I llama.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

#
# Build library
#

ggml.o: ggml.c ggml.h ggml-cuda.h
	$(CC)  $(CFLAGS)   -c $< -o $@

ggml-alloc.o: ggml-alloc.c ggml.h ggml-alloc.h
	$(CC)  $(CFLAGS)   -c $< -o $@

OBJS += ggml-alloc.o

llama.o: llama.cpp ggml.h ggml-alloc.h ggml-cuda.h ggml-metal.h llama.h llama-util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

common.o: examples/common.cpp examples/common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

console.o: examples/console.cpp examples/console.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

grammar-parser.o: examples/grammar-parser.cpp examples/grammar-parser.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

libllama.so: llama.o ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

clean:
	rm -vf *.o *.so *.dll main quantize quantize-stats perplexity embedding benchmark-matmult save-load-state server simple vdot train-text-from-scratch convert-llama2c-to-ggml embd-input-test build-info.h $(TEST_TARGETS)

#
# Examples
#

main: examples/main/main.cpp                                  build-info.h ggml.o llama.o common.o console.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./main -h for help.  ===='
	@echo

simple: examples/simple/simple.cpp                            build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

quantize: examples/quantize/quantize.cpp                      build-info.h ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

quantize-stats: examples/quantize-stats/quantize-stats.cpp    build-info.h ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

perplexity: examples/perplexity/perplexity.cpp                build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

embedding: examples/embedding/embedding.cpp                   build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

save-load-state: examples/save-load-state/save-load-state.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

server: examples/server/server.cpp examples/server/httplib.h examples/server/json.hpp examples/server/index.html.hpp examples/server/index.js.hpp examples/server/completion.js.hpp build-info.h ggml.o llama.o common.o grammar-parser.o $(OBJS)
	$(CXX) $(CXXFLAGS) -Iexamples/server $(filter-out %.h,$(filter-out %.hpp,$^)) -o $@ $(LDFLAGS) $(LWINSOCK2)

$(LIB_PRE)embdinput$(DSO_EXT): examples/embd-input/embd-input.h examples/embd-input/embd-input-lib.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) --shared $(CXXFLAGS) $(filter-out %.h,$(filter-out %.hpp,$^)) -o $@ $(LDFLAGS)


embd-input-test: $(LIB_PRE)embdinput$(DSO_EXT) examples/embd-input/embd-input-test.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %$(DSO_EXT),$(filter-out %.h,$(filter-out %.hpp,$^))) -o $@ $(LDFLAGS) -L. -lembdinput

train-text-from-scratch: examples/train-text-from-scratch/train-text-from-scratch.cpp    build-info.h ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

convert-llama2c-to-ggml: examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp    build-info.h ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)

build-info.h: $(wildcard .git/index) scripts/build-info.sh
	@sh scripts/build-info.sh > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
		mv $@.tmp $@; \
	else \
		rm $@.tmp; \
	fi

#
# Tests
#

tests: $(TEST_TARGETS)

benchmark-matmult: examples/benchmark/benchmark-matmult.cpp build-info.h ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	./$@

vdot: pocs/vdot/vdot.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

tests/test-grammar-parser: tests/test-grammar-parser.cpp examples/grammar-parser.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-double-float: tests/test-double-float.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-grad0: tests/test-grad0.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-opt: tests/test-opt.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-quantize-fns: tests/test-quantize-fns.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-quantize-perf: tests/test-quantize-perf.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-sampling: tests/test-sampling.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)

tests/test-tokenizer-0: tests/test-tokenizer-0.cpp build-info.h ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.txt,$^) -o $@ $(LDFLAGS)
