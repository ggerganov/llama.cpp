default: koboldcpp koboldcpp_failsafe koboldcpp_openblas koboldcpp_openblas_noavx2 koboldcpp_clblast koboldcpp_cublas
tools: quantize_gpt2 quantize_gptj quantize_llama quantize_neox quantize_mpt
dev: koboldcpp_openblas
dev2: koboldcpp_clblast


ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

ifneq ($(shell grep -e "Arch Linux" -e "ID_LIKE=arch" /etc/os-release 2>/dev/null),)
ARCH_ADD = -lcblas
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
CFLAGS   = -I.              -I./include -I./include/CL -I./otherarch -I./otherarch/tools -Ofast -DNDEBUG -std=c11   -fPIC -DGGML_USE_K_QUANTS
CXXFLAGS = -I. -I./examples -I./include -I./include/CL -I./otherarch -I./otherarch/tools -O3 -DNDEBUG -std=c++11 -fPIC
LDFLAGS  =

# these are used on windows, to build some libraries with extra old device compatibility
SIMPLECFLAGS =
FULLCFLAGS =
NONECFLAGS =

OPENBLAS_FLAGS = -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
CLBLAST_FLAGS = -DGGML_USE_CLBLAST
FAILSAFE_FLAGS = -DUSE_FAILSAFE
CUBLAS_FLAGS =
CUBLASLD_FLAGS =
CUBLAS_OBJS =

#lets try enabling everything
CFLAGS   += -pthread -s
CXXFLAGS += -pthread -s -Wno-multichar -Wno-write-strings

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
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
# old library NEEDS mf16c to work. so we must build with it. new one doesnt
	ifeq ($(OS),Windows_NT)
		CFLAGS +=
		NONECFLAGS += -mno-sse3
		SIMPLECFLAGS += -mavx -msse3
		FULLCFLAGS += -mavx2 -msse3 -mfma -mf16c -mavx
	else
# if not on windows, they are clearly building it themselves, so lets just use whatever is supported
		CFLAGS += -march=native -mtune=native
	endif
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
ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif

# it is recommended to use the CMAKE file to build for cublas if you can - will likely work better
ifdef LLAMA_CUBLAS
	CUBLAS_FLAGS = -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I$(CUDA_PATH)/targets/x86_64-linux/include
	CUBLASLD_FLAGS = -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib
	CUBLAS_OBJS = ggml-cuda.o ggml_v2-cuda.o ggml_v2-cuda-legacy.o
	NVCC      = nvcc
	NVCCFLAGS = --forward-unknown-to-host-compiler -arch=native
ifdef LLAMA_CUDA_DMMV_X
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=$(LLAMA_CUDA_DMMV_X)
else
	NVCCFLAGS += -DGGML_CUDA_DMMV_X=32
endif # LLAMA_CUDA_DMMV_X
ifdef LLAMA_CUDA_DMMV_Y
	NVCCFLAGS += -DGGML_CUDA_DMMV_Y=$(LLAMA_CUDA_DMMV_Y)
else
	NVCCFLAGS += -DGGML_CUDA_DMMV_Y=1
endif # LLAMA_CUDA_DMMV_Y
ifdef LLAMA_CUDA_DMMV_F16
	NVCCFLAGS += -DGGML_CUDA_DMMV_F16
endif # LLAMA_CUDA_DMMV_F16
ifdef LLAMA_CUDA_KQUANTS_ITER
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=$(LLAMA_CUDA_KQUANTS_ITER)
else
	NVCCFLAGS += -DK_QUANTS_PER_ITERATION=2
endif
ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v2-cuda.o: otherarch/ggml_v2-cuda.cu otherarch/ggml_v2-cuda.h
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
ggml_v2-cuda-legacy.o: otherarch/ggml_v2-cuda-legacy.cu otherarch/ggml_v2-cuda-legacy.h
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(CUBLAS_FLAGS) $(CUBLAS_CXXFLAGS) -Wno-pedantic -c $< -o $@
endif # LLAMA_CUBLAS

ifdef LLAMA_METAL
	CFLAGS   += -DGGML_USE_METAL -DGGML_METAL_NDEBUG
	CXXFLAGS += -DGGML_USE_METAL
	LDFLAGS  += -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
	OBJS     += ggml-metal.o

ggml-metal.o: ggml-metal.m ggml-metal.h
	$(CC) $(CFLAGS) -c $< -o $@
endif # LLAMA_METAL

ifneq ($(filter aarch64%,$(UNAME_M)),)
	# Apple M1, M2, etc.
	# Raspberry Pi 3, 4, Zero 2 (64-bit)
	CFLAGS +=
	CXXFLAGS +=
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

DEFAULT_BUILD =
FAILSAFE_BUILD =
OPENBLAS_BUILD =
OPENBLAS_NOAVX2_BUILD =
CLBLAST_BUILD =
CUBLAS_BUILD =

ifeq ($(OS),Windows_NT)
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.dll $(LDFLAGS)
	FAILSAFE_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.dll $(LDFLAGS)
	OPENBLAS_BUILD = $(CXX) $(CXXFLAGS) $^ lib/libopenblas.lib -shared -o $@.dll $(LDFLAGS)
	OPENBLAS_NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ lib/libopenblas.lib -shared -o $@.dll $(LDFLAGS)
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -shared -o $@.dll $(LDFLAGS)

ifdef LLAMA_CUBLAS
	CUBLAS_BUILD = $(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $^ -shared -o $@.dll $(CUBLASLD_FLAGS) $(LDFLAGS)
endif

else
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
	FAILSAFE_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.so $(LDFLAGS)
	ifdef LLAMA_OPENBLAS
	OPENBLAS_BUILD = $(CXX) $(CXXFLAGS) $^ $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
	OPENBLAS_NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
	endif
	ifdef LLAMA_CLBLAST
        ifeq ($(UNAME_S),Darwin)
                CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -framework OpenCL $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
        else
                CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -lOpenCL $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
        endif
	endif

ifdef LLAMA_CUBLAS
	CUBLAS_BUILD = $(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) $^ -shared -o $@.so $(CUBLASLD_FLAGS) $(LDFLAGS)
endif

	ifndef LLAMA_OPENBLAS
	ifndef LLAMA_CLBLAST
	ifndef LLAMA_CUBLAS
	OPENBLAS_BUILD = @echo 'Your OS $(OS) does not appear to be Windows. For faster speeds, install and link a BLAS library. Set LLAMA_OPENBLAS=1 to compile with OpenBLAS support or LLAMA_CLBLAST=1 to compile with ClBlast support. This is just a reminder, not an error.'
	endif
	endif
	endif
endif



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

ggml.o: ggml.c ggml.h ggml-cuda.h k_quants.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_openblas.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
ggml_failsafe.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@
ggml_openblas_noavx2.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(SIMPLECFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
ggml_clblast.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_cublas.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) -c $< -o $@

#quants K
k_quants.o: k_quants.c k_quants.h ggml.h ggml-cuda.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
k_quants_noavx2.o: k_quants.c k_quants.h ggml.h ggml-cuda.h
	$(CC)  $(CFLAGS) $(SIMPLECFLAGS) -c $< -o $@
k_quants_failsafe.o: k_quants.c k_quants.h ggml.h ggml-cuda.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@

#version 2 libs
ggml_v2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v2_openblas.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
ggml_v2_failsafe.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@
ggml_v2_openblas_noavx2.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(CFLAGS) $(SIMPLECFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@
ggml_v2_clblast.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2_cublas.o: otherarch/ggml_v2.c otherarch/ggml_v2.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) $(CUBLAS_FLAGS) -c $< -o $@

#extreme old version compat
ggml_v1.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(CFLAGS) $(FULLCFLAGS) -c $< -o $@
ggml_v1_failsafe.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(CFLAGS) $(NONECFLAGS) -c $< -o $@

#opencl
ggml-opencl.o: ggml-opencl.cpp ggml-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2-opencl.o: otherarch/ggml_v2-opencl.cpp otherarch/ggml_v2-opencl.h
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
ggml_v2-opencl-legacy.o: otherarch/ggml_v2-opencl-legacy.c otherarch/ggml_v2-opencl-legacy.h
	$(CC) $(CFLAGS) -c $< -o $@

# intermediate objects
llama.o: llama.cpp ggml.h ggml-cuda.h llama.h llama-util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
common.o: examples/common.cpp examples/common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
expose.o: expose.cpp expose.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
gpttype_adapter_failsafe.o: gpttype_adapter.cpp
	$(CXX) $(CXXFLAGS) $(FAILSAFE_FLAGS) -c $< -o $@
gpttype_adapter.o: gpttype_adapter.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
gpttype_adapter_clblast.o: gpttype_adapter.cpp
	$(CXX) $(CXXFLAGS) $(CLBLAST_FLAGS) -c $< -o $@
gpttype_adapter_cublas.o: gpttype_adapter.cpp
	$(CXX) $(CXXFLAGS) $(CUBLAS_FLAGS) -c $< -o $@

clean:
	rm -vf *.o main quantize_llama quantize_gpt2 quantize_gptj quantize_neox quantize_mpt quantize-stats perplexity embedding benchmark-matmult save-load-state main.exe quantize_llama.exe quantize_gptj.exe quantize_gpt2.exe quantize_neox.exe quantize_mpt.exe koboldcpp.dll koboldcpp_openblas.dll koboldcpp_failsafe.dll koboldcpp_openblas_noavx2.dll koboldcpp_clblast.dll koboldcpp_cublas.dll koboldcpp.so koboldcpp_openblas.so koboldcpp_failsafe.so koboldcpp_openblas_noavx2.so koboldcpp_clblast.so koboldcpp_cublas.so

main: examples/main/main.cpp build-info.h ggml.o k_quants.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./main -h for help.  ===='
	@echo

#generated libraries
koboldcpp: ggml.o ggml_v2.o ggml_v1.o expose.o common.o gpttype_adapter.o k_quants.o $(OBJS)
	$(DEFAULT_BUILD)
koboldcpp_openblas: ggml_openblas.o ggml_v2_openblas.o ggml_v1.o expose.o common.o gpttype_adapter.o k_quants.o $(OBJS)
	$(OPENBLAS_BUILD)
koboldcpp_failsafe: ggml_failsafe.o ggml_v2_failsafe.o ggml_v1_failsafe.o expose.o common.o gpttype_adapter_failsafe.o k_quants_failsafe.o $(OBJS)
	$(FAILSAFE_BUILD)
koboldcpp_openblas_noavx2: ggml_openblas_noavx2.o ggml_v2_openblas_noavx2.o ggml_v1_failsafe.o expose.o common.o gpttype_adapter.o k_quants_noavx2.o $(OBJS)
	$(OPENBLAS_NOAVX2_BUILD)
koboldcpp_clblast: ggml_clblast.o ggml_v2_clblast.o ggml_v1.o expose.o common.o gpttype_adapter_clblast.o ggml-opencl.o ggml_v2-opencl.o ggml_v2-opencl-legacy.o k_quants.o $(OBJS)
	$(CLBLAST_BUILD)
koboldcpp_cublas: ggml_cublas.o ggml_v2_cublas.o ggml_v1.o expose.o common.o gpttype_adapter_cublas.o k_quants.o $(CUBLAS_OBJS) $(OBJS)
	$(CUBLAS_BUILD)

quantize_llama: examples/quantize/quantize.cpp ggml.o llama.o k_quants.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gptj: ggml.o llama.o k_quants.o otherarch/tools/gptj_quantize.cpp otherarch/tools/common-ggml.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_gpt2: ggml.o llama.o k_quants.o otherarch/tools/gpt2_quantize.cpp otherarch/tools/common-ggml.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_neox: ggml.o llama.o k_quants.o otherarch/tools/neox_quantize.cpp otherarch/tools/common-ggml.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
quantize_mpt: ggml.o llama.o k_quants.o otherarch/tools/mpt_quantize.cpp otherarch/tools/common-ggml.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


build-info.h:
	$(DONOTHING)
