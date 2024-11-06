#ifndef GPTSampler_h
#define GPTSampler_h

@class LlamaModel;
@class GPTSamplerParams;
@class LlamaContext;
typedef int32_t LlamaToken;

@interface GPTSampler : NSObject

- (instancetype)init:(LlamaModel *)model gptSamplerParams:(GPTSamplerParams *)gptSamplerParams;
- (uint32_t)seed;

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
- (LlamaToken)sample:(LlamaContext *)context
               index:(NSInteger) index;

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
- (LlamaToken)sample:(LlamaContext *)context
               index:(NSInteger) index
        grammarFirst:(BOOL)grammarFirst;

/// If accept_grammar is true, the token is accepted both by the sampling chain and the grammar
- (void)accept:(LlamaToken)token
 acceptGrammar:(BOOL)acceptGrammar;

/// Get a string representation of the last accepted tokens
- (NSString *)previousString:(LlamaContext *)context n:(NSInteger)n;

/// Get the last accepted token
- (LlamaToken)last;

- (void)reset;

@end

#endif /* GPTSampler_h */
