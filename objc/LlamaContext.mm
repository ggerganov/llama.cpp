#import <Foundation/Foundation.h>
#import "LlamaContext_Private.hpp"
#import "GGMLThreadpool_Private.hpp"
#import "GPTParams_Private.hpp"
#import "LlamaModel_Private.hpp"
#import "LlamaBatch_Private.hpp"
#import "../../common/common.h"

@implementation LlamaContext {
    llama_context *ctx;
}

- (instancetype)initWithContext:(llama_context *)context {
    self = [super init];
    if (self) {
        ctx = context;
    }
    return self;
}

- (void)dealloc
{
    llama_free(ctx);
    [super dealloc];
}

- (void)attachThreadpool:(GGMLThreadpool *)threadpool
         threadpoolBatch:(GGMLThreadpool *)threadpoolBatch {
    llama_attach_threadpool(ctx, [threadpool threadpool], [threadpoolBatch threadpool]);
}


- (NSUInteger)nCtx {
    return llama_n_ctx(ctx);
}

- (BOOL)loadStateFile:(NSString *)pathSession
            tokensOut:(llama_token *)tokensOut
        nTokenCpacity:(size_t)nTokenCapacity
       nTokenCountOut:(size_t *)nTokenCountOut {
    return llama_state_load_file(ctx, [pathSession cStringUsingEncoding:NSUTF8StringEncoding], tokensOut, nTokenCapacity, nTokenCountOut);
}

- (LlamaModel *)model {
    auto model = llama_get_model(ctx);
    return [[LlamaModel alloc] init:std::remove_const_t<llama_model *>(model)];
}

- (std::vector<llama_token>)tokenize:(NSString *)text
addSpecial:(BOOL)addSpecial
parseSpecial:(BOOL)parseSpecial {
    
    return common_tokenize(ctx, [text cStringUsingEncoding:NSUTF8StringEncoding], addSpecial, parseSpecial);
}

- (std::string)convertTokensToString:(const std::vector<llama_token>&)tokens {
    return string_from(ctx, tokens);
}

- (llama_context *)cContext {
    return ctx;
}

- (int32_t)encode:(llama_batch)batch {
    return llama_encode(ctx, batch);
}

- (void)kvCacheSeqAdd:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta {
    llama_kv_cache_seq_add(ctx, sequenceId, p0, p1, delta);
}

- (void)kvCacheSeqDiv:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta {
    llama_kv_cache_seq_div(ctx, sequenceId, p0, p1, delta);
}

- (NSString *)tokenToPiece:(LlamaToken)token {
    return [self tokenToPiece:token special:YES];
}

- (NSString *)tokenToPiece:(LlamaToken)token special:(BOOL)special {
    return [[NSString alloc] initWithCString:common_token_to_piece(ctx, token, special).c_str() encoding:NSUTF8StringEncoding];
}

- (NSInteger)decode:(LlamaBatch *)batch {
    return llama_decode(ctx, [batch cBatch]);
}

- (BOOL)saveStateFile:(NSString *)pathSession
               tokens:(const LlamaToken *)tokens
          nTokenCount:(size_t)nTokenCount {
    return llama_state_save_file(ctx,
                                 [pathSession cStringUsingEncoding:NSUTF8StringEncoding],
                                 tokens, nTokenCount);
}

@end
