#ifndef LlamaBatch_Private_hpp
#define LlamaBatch_Private_hpp
#import "LlamaBatch.h"
#import "llama.h"

@interface LlamaBatch()

- (instancetype)initWithBatch:(llama_batch)batch;
- (llama_batch)cBatch;

@end

#endif /* LlamaBatch_Private_hpp */
