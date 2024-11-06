#ifndef LlamaObjC_h
#define LlamaObjC_h

#include <Foundation/Foundation.h>
#include <CPUParams.h>
#include <GGMLThreadpool.h>
#include <GPTParams.h>
#include <GPTSampler.h>
#include <llama.h>
#include <LlamaBatch.h>
#include <LlamaContext.h>
#include <LlamaModel.h>
#include <LlamaSession.h>

int LLAMA_BUILD_NUMBER = 0;
char const * LLAMA_COMMIT = "unknown";
char const * LLAMA_COMPILER = "unknown";
char const * LLAMA_BUILD_TARGET = "unknown";

#endif /* LlamaObjC_h */
