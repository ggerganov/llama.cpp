#import <Foundation/Foundation.h>
#import <GPTSampler.h>
#import <GPTParams_Private.hpp>
#import <LlamaModel_Private.hpp>
#import <LlamaContext_Private.hpp>
#import "../../common/sampling.h"

@implementation GPTSampler {
    common_sampler *sampler;
}

- (instancetype)init:(LlamaModel *)model gptSamplerParams:(GPTSamplerParams *)gptSamplerParams
{
    self = [super init];
    if (self) {
        self->sampler = common_sampler_init([model cModel], [gptSamplerParams cParams]);
    }
    return self;
}

- (uint32_t)seed {
    return common_sampler_get_seed(sampler);
}

- (LlamaToken)sample:(LlamaContext *)context index:(NSInteger)index {
    return [self sample:context index:index grammarFirst:false];
}

- (LlamaToken)sample:(LlamaContext *)context index:(NSInteger)index grammarFirst:(BOOL)grammarFirst {
    return common_sampler_sample(sampler, [context cContext], index, grammarFirst);
}

- (void)accept:(LlamaToken)token acceptGrammar:(BOOL)acceptGrammar {
    common_sampler_accept(sampler, token, acceptGrammar);
}

- (NSString *)previousString:(LlamaContext *)context n:(NSInteger)n {
    return [[NSString alloc] initWithCString:common_sampler_prev_str(sampler, [context cContext], n).data() encoding:NSUTF8StringEncoding];
}

- (LlamaToken)last {
    return common_sampler_last(sampler);
}

- (void)reset {
    common_sampler_reset(sampler);
}

- (void)dealloc
{
    common_sampler_free(sampler);
    [super dealloc];
}

@end
