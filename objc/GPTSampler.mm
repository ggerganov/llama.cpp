#import <Foundation/Foundation.h>
#import <GPTSampler.h>
#import <GPTParams_Private.hpp>
#import <LlamaModel_Private.hpp>
#import <LlamaContext_Private.hpp>
#import "../../common/sampling.h"

@implementation GPTSampler {
    gpt_sampler *sampler;
}

- (instancetype)init:(LlamaModel *)model gptSamplerParams:(GPTSamplerParams *)gptSamplerParams
{
    self = [super init];
    if (self) {
        self->sampler = gpt_sampler_init([model cModel], [gptSamplerParams cParams]);
    }
    return self;
}

- (uint32_t)seed {
    return gpt_sampler_get_seed(sampler);
}

- (LlamaToken)sample:(LlamaContext *)context index:(NSInteger)index {
    return [self sample:context index:index grammarFirst:false];
}

- (LlamaToken)sample:(LlamaContext *)context index:(NSInteger)index grammarFirst:(BOOL)grammarFirst {
    return gpt_sampler_sample(sampler, [context cContext], index, grammarFirst);
}

- (void)accept:(LlamaToken)token acceptGrammar:(BOOL)acceptGrammar {
    gpt_sampler_accept(sampler, token, acceptGrammar);
}

- (NSString *)previousString:(LlamaContext *)context n:(NSInteger)n {
    return [[NSString alloc] initWithCString:gpt_sampler_prev_str(sampler, [context cContext], n).data() encoding:NSUTF8StringEncoding];
}

- (LlamaToken)last {
    return gpt_sampler_last(sampler);
}

- (void)reset {
    gpt_sampler_reset(sampler);
}

@end
