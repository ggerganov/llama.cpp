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
CFLAGS   = -I.              -O3 -DNDEBUG -std=c11   -fPIC
CXXFLAGS = -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC
LDFLAGS  =

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
	ifeq ($(UNAME_S),Darwin)
		CFLAGS += -mf16c
		AVX1_M := $(shell sysctl machdep.cpu.features)
		ifneq (,$(findstring FMA,$(AVX1_M)))
			CFLAGS += -mfma
		endif
		ifneq (,$(findstring AVX1.0,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell sysctl machdep.cpu.leaf7_features)
		ifneq (,$(findstring AVX2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
	else ifeq ($(UNAME_S),Linux)
		AVX1_M := $(shell grep "avx " /proc/cpuinfo)
		ifneq (,$(findstring avx,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell grep "avx2 " /proc/cpuinfo)
		ifneq (,$(findstring avx2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell grep "fma " /proc/cpuinfo)
		ifneq (,$(findstring fma,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell grep "f16c " /proc/cpuinfo)
		ifneq (,$(findstring f16c,$(F16C_M)))
			CFLAGS += -mf16c
		endif
		SSE3_M := $(shell grep "sse3 " /proc/cpuinfo)
		ifneq (,$(findstring sse3,$(SSE3_M)))
			CFLAGS += -msse3
		endif
		AVX512F_M := $(shell grep "avx512f " /proc/cpuinfo)
		ifneq (,$(findstring avx512f,$(AVX512F_M)))
			CFLAGS += -mavx512f
		endif
		AVX512BW_M := $(shell grep "avx512bw " /proc/cpuinfo)
		ifneq (,$(findstring avx512bw,$(AVX512BW_M)))
			CFLAGS += -mavx512bw
		endif
		AVX512DQ_M := $(shell grep "avx512dq " /proc/cpuinfo)
		ifneq (,$(findstring avx512dq,$(AVX512DQ_M)))
			CFLAGS += -mavx512dq
		endif
		AVX512VL_M := $(shell grep "avx512vl " /proc/cpuinfo)
		ifneq (,$(findstring avx512vl,$(AVX512VL_M)))
			CFLAGS += -mavx512vl
		endif
		AVX512CD_M := $(shell grep "avx512cd " /proc/cpuinfo)
		ifneq (,$(findstring avx512cd,$(AVX512CD_M)))
			CFLAGS += -mavx512cd
		endif
		AVX512ER_M := $(shell grep "avx512er " /proc/cpuinfo)
		ifneq (,$(findstring avx512er,$(AVX512ER_M)))
			CFLAGS += -mavx512er
		endif
		AVX512IFMA_M := $(shell grep "avx512ifma " /proc/cpuinfo)
		ifneq (,$(findstring avx512ifma,$(AVX512IFMA_M)))
			CFLAGS += -mavx512ifma
		endif
		AVX512PF_M := $(shell grep "avx512pf " /proc/cpuinfo)
		ifneq (,$(findstring avx512pf,$(AVX512PF_M)))
			CFLAGS += -mavx512pf
		endif
	else ifeq ($(UNAME_S),Haiku)
		AVX1_M := $(shell sysinfo -cpu | grep -w "AVX")
		ifneq (,$(findstring AVX,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell sysinfo -cpu | grep -w "AVX2")
		ifneq (,$(findstring AVX2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell sysinfo -cpu | grep -w "FMA")
		ifneq (,$(findstring FMA,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell sysinfo -cpu | grep -w "F16C")
		ifneq (,$(findstring F16C,$(F16C_M)))
			CFLAGS += -mf16c
		endif
	else
		CFLAGS += -mfma -mf16c -mavx -mavx2
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
ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
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

default: main quantize

#
# Build library
#

ggml.o: ggml.c ggml.h
	$(CC)  $(CFLAGS)   -c ggml.c -o ggml.o

llama.o: llama.cpp llama.h
	$(CXX) $(CXXFLAGS) -c llama.cpp -o llama.o

utils.o: utils.cpp utils.h
	$(CXX) $(CXXFLAGS) -c utils.cpp -o utils.o

clean:
	rm -f *.o main quantize

main: main.cpp ggml.o llama.o utils.o
	$(CXX) $(CXXFLAGS) main.cpp ggml.o llama.o utils.o -o main $(LDFLAGS)
	@echo
	@echo '====  Run ./main -h for help.  ===='
	@echo

quantize: quantize.cpp ggml.o llama.o utils.o
	$(CXX) $(CXXFLAGS) quantize.cpp ggml.o llama.o utils.o -o quantize $(LDFLAGS)

#
# Tests
#

.PHONY: tests
tests:
	bash ./tests/run-tests.sh
