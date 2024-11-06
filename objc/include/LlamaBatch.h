#ifndef LlamaBatch_h
#define LlamaBatch_h

typedef NSInteger LlamaSequenceId;
typedef NSInteger LlamaPosition;
typedef int32_t LlamaToken;

/// Input data for llama_decode
/// A llama_batch object can contain input about one or many sequences
/// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
///
/// - token  : the token ids of the input (used when embd is NULL)
/// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
/// - pos    : the positions of the respective token in the sequence
/// - seq_id : the sequence to which the respective token belongs
/// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
@interface LlamaBatch : NSObject

@property (nonatomic, assign) NSInteger nTokens;
@property (nonatomic, assign) LlamaToken *tokens;
@property (nonatomic, assign) float *embd;
@property (nonatomic, assign) LlamaPosition *pos;
@property (nonatomic, assign) int32_t *nSeqId;
@property (nonatomic, assign) LlamaSequenceId **seqId;
@property (nonatomic, assign) NSData *output;

// Helpers for smooth API transition (optional usage in the interface)
@property (nonatomic, assign) LlamaPosition allPos0;
@property (nonatomic, assign) LlamaPosition allPos1;
@property (nonatomic, assign) LlamaSequenceId allSeqId;

@end

#endif /* LlamaBatch_h */
