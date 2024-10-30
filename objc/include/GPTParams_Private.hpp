#ifndef GPTParams_Private_hpp
#define GPTParams_Private_hpp

#import "GPTParams.h"
#import "ggml.h"
#import "../../common/common.h"

@interface GGMLThreadpool()

- (ggml_threadpool *)threadpool;

@end

@interface GPTParams()

- (gpt_params&)params;

@end

@interface GPTSamplerParams()

- (gpt_sampler_params&)cParams;

@end
#endif /* GPTParams_Private_hpp */
