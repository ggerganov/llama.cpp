#import <Foundation/Foundation.h>
#import "LlamaBatch_Private.hpp"
#import "llama.h"

@implementation LlamaBatch {
    llama_batch batch;
}

- (instancetype)initWithBatch:(llama_batch)batch {
    self->batch = batch;
}

- (NSData *)output {
    return [[NSData alloc] initWithBytes:batch.logits length:batch.n_tokens];
}

- (llama_batch)cBatch {
    return batch;
}

@end
