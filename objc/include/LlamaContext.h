#ifndef LlamaContext_h
#define LlamaContext_h

@class GGMLThreadpool;
@class LlamaBatch;

typedef NSInteger LlamaSequenceId;
typedef NSInteger LlamaPosition;
typedef int32_t LlamaToken;

@interface LlamaContext : NSObject

- (NSUInteger)nCtx;

- (void)attachThreadpool:(GGMLThreadpool *)threadpool
         threadpoolBatch:(GGMLThreadpool *)threadpoolBatch;

// Positive return values does not mean a fatal error, but rather a warning.
//   0 - success
//   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
// < 0 - error
- (NSInteger)decode:(LlamaBatch *)batch;

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// If the KV cache is RoPEd, the KV data is updated accordingly:
//   - lazily on next llama_decode()
//   - explicitly with llama_kv_cache_update()
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
- (void)kvCacheSeqAdd:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta;

// Integer division of the positions by factor of `d > 1`
// If the KV cache is RoPEd, the KV data is updated accordingly:
//   - lazily on next llama_decode()
//   - explicitly with llama_kv_cache_update()
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
- (void)kvCacheSeqDiv:(LlamaSequenceId)sequenceId
                   p0:(LlamaPosition)p0
                   p1:(LlamaPosition)p1
                delta:(LlamaPosition)delta;

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
- (NSString *)tokenToPiece:(LlamaToken)token;
- (NSString *)tokenToPiece:(LlamaToken)token special:(BOOL)special;

- (BOOL)saveStateFile:(NSString *)pathSession
               tokens:(const LlamaToken *)tokens
          nTokenCount:(size_t)nTokenCount;

@end

#endif /* LlamaContext_h */
