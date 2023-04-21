default: koboldcpp koboldcpp_noavx2 koboldcpp_openblas koboldcpp_openblas_noavx2 koboldcpp_clblast
simple: koboldcpp koboldcpp_noavx2
dev: koboldcpp_openblas


ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

ARCH_LINUX1 := $(shell grep "Arch Linux" /etc/os-release 2>/dev/null)
ARCH_LINUX2 := $(shell grep "ID_LIKE=arch" /etc/os-release 2>/dev/null)
ifdef ARCH_LINUX1
ARCH_ADD = -lcblas
endif
ifdef ARCH_LINUX2
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
CFLAGS   = -I.              -I./include -I./include/CL -Ofast -DNDEBUG -std=c11   -fPIC
CXXFLAGS = -I. -I./examples -I./include -I./include/CL -Ofast -DNDEBUG -std=c++11 -fPIC
LDFLAGS  =

# these are used on windows, to build some libraries with extra old device compatibility
BONUSCFLAGS1 =
BONUSCFLAGS2 =

OPENBLAS_FLAGS = -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
CLBLAST_FLAGS = -DGGML_USE_CLBLAST -DGGML_USE_OPENBLAS -I/usr/local/include/openblas

#lets try enabling everything
CFLAGS   += -pthread -s
CXXFLAGS += -pthread -s -Wno-multichar

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

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
	CFLAGS += -mavx
# old library NEEDS mf16c to work. so we must build with it. new one doesnt
	ifeq ($(OS),Windows_NT)
		BONUSCFLAGS1 += -mf16c
		BONUSCFLAGS2 += -mavx2 -msse3 -mfma
	else
# if not on windows, they are clearly building it themselves, so lets just use whatever is supported
		CFLAGS += -march=native -mtune=native
	endif
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mcpu=power9
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

#note: koboldcpp does not officially support cublas. You can manually link it if you want, but it will not be a regular feature
ifdef LLAMA_CUBLAS
	CFLAGS  += -DGGML_USE_CUBLAS -I/usr/local/cuda/include
	LDFLAGS += -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64
	OBJS	+= ggml-cuda.o
ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
	nvcc -arch=native -c -o $@ $<
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS +=
	CXXFLAGS +=
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, 2, 3
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

DEFAULT_BUILD =
NOAVX2_BUILD =
OPENBLAS_BUILD =
OPENBLAS_NOAVX2_BUILD =
CLBLAST_BUILD =

ifeq ($(OS),Windows_NT)
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.dll $(LDFLAGS)
	NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.dll $(LDFLAGS)
	OPENBLAS_BUILD = $(CXX) $(CXXFLAGS) $^ lib/libopenblas.lib -shared -o $@.dll $(LDFLAGS)
	OPENBLAS_NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ lib/libopenblas.lib -shared -o $@.dll $(LDFLAGS)
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ lib/OpenCL.lib lib/clblast.lib -shared -o $@.dll $(LDFLAGS)
else
	DEFAULT_BUILD = $(CXX) $(CXXFLAGS)  $^ -shared -o $@.so $(LDFLAGS)
	NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ -shared -o $@.so $(LDFLAGS)
	ifdef LLAMA_OPENBLAS
	OPENBLAS_BUILD = $(CXX) $(CXXFLAGS) $^ $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
	OPENBLAS_NOAVX2_BUILD = $(CXX) $(CXXFLAGS) $^ $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
	endif	
	ifdef LLAMA_CLBLAST
	CLBLAST_BUILD = $(CXX) $(CXXFLAGS) $^ -lclblast -lOpenCL $(ARCH_ADD) -lopenblas -shared -o $@.so $(LDFLAGS)
	endif

	ifndef LLAMA_OPENBLAS
	ifndef LLAMA_CLBLAST
	OPENBLAS_BUILD = @echo 'Your OS $(OS) does not appear to be Windows. For faster speeds, install and link a BLAS library. Set LLAMA_OPENBLAS=1 to compile with OpenBLAS support or LLAMA_CLBLAST=1 to compile with ClBlast support. This is just a reminder, not an error.'
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

ggml.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(BONUSCFLAGS1) $(BONUSCFLAGS2) -c $< -o $@

ggml_openblas.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(BONUSCFLAGS1) $(BONUSCFLAGS2) $(OPENBLAS_FLAGS) -c $< -o $@

ggml_noavx2.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) -c $< -o $@

ggml_openblas_noavx2.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(OPENBLAS_FLAGS) -c $< -o $@

ggml_clblast.o: ggml.c ggml.h
	$(CC)  $(CFLAGS) $(BONUSCFLAGS1) $(BONUSCFLAGS2) $(CLBLAST_FLAGS) -c $< -o $@

ggml_v1.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(CFLAGS) $(BONUSCFLAGS1) $(BONUSCFLAGS2) -c $< -o $@

ggml_v1_noavx2.o: otherarch/ggml_v1.c otherarch/ggml_v1.h
	$(CC)  $(CFLAGS) $(BONUSCFLAGS1) -c $< -o $@

ggml_rwkv.o: otherarch/ggml_rwkv.c otherarch/ggml_rwkv.h
	$(CC)  $(CFLAGS) $(BONUSCFLAGS1) $(BONUSCFLAGS2) -c $< -o $@

llama.o: llama.cpp llama.h llama_util.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

common.o: examples/common.cpp examples/common.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

expose.o: expose.cpp expose.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

gpttype_adapter.o: gpttype_adapter.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -vf *.o main quantize_llama quantize_gpt2 quantize_gptj quantize-stats perplexity embedding benchmark-q4_0-matmult main.exe quantize_llama.exe quantize_gptj.exe quantize_gpt2.exe koboldcpp.dll koboldcpp_openblas.dll koboldcpp_noavx2.dll koboldcpp_openblas_noavx2.dll koboldcpp_clblast.dll koboldcpp.so koboldcpp_openblas.so koboldcpp_noavx2.so koboldcpp_openblas_noavx2.so koboldcpp_clblast.so gptj.exe gpt2.exe

main: examples/main/main.cpp ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo
	@echo '====  Run ./main -h for help.  ===='
	@echo

koboldcpp: ggml.o ggml_rwkv.o ggml_v1.o expose.o common.o gpttype_adapter.o
	$(DEFAULT_BUILD)

koboldcpp_openblas: ggml_openblas.o ggml_rwkv.o ggml_v1.o expose.o common.o gpttype_adapter.o 
	$(OPENBLAS_BUILD)
	
koboldcpp_noavx2: ggml_noavx2.o ggml_rwkv.o ggml_v1_noavx2.o expose.o common.o gpttype_adapter.o 
	$(NOAVX2_BUILD)

koboldcpp_openblas_noavx2: ggml_openblas_noavx2.o ggml_rwkv.o ggml_v1_noavx2.o expose.o common.o gpttype_adapter.o 
	$(OPENBLAS_NOAVX2_BUILD)

koboldcpp_clblast: ggml_clblast.o ggml_rwkv.o ggml_v1.o expose.o common.o gpttype_adapter.o 
	$(CLBLAST_BUILD)
		
quantize_llama: examples/quantize/quantize.cpp ggml.o llama.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

quantize_gptj: ggml.o llama.o otherarch/tools/gptj_quantize.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

quantize_gpt2: ggml.o llama.o otherarch/tools/gpt2_quantize.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

quantize-stats: examples/quantize-stats/quantize-stats.cpp ggml.o llama.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

perplexity: examples/perplexity/perplexity.cpp ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

embedding: examples/embedding/embedding.cpp ggml.o llama.o common.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

vdot: pocs/vdot/vdot.cpp ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

libllama.so: llama.o ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

#
# Tests
#

benchmark: examples/benchmark/benchmark-q4_0-matmult.c ggml.o $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o benchmark-q4_0-matmult $(LDFLAGS)
	./benchmark-q4_0-matmult

.PHONY: tests
tests:
	bash ./tests/run-tests.sh
