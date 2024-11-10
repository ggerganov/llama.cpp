#import <Foundation/Foundation.h>
#import "LlamaModel_Private.hpp"
#import "LlamaContext_Private.hpp"
#import "LlamaBatch_Private.hpp"
#import "GPTParams_Private.hpp"
#import "GPTSampler.h"
#import "ggml.h"
#import "../common/common.h"

@implementation LlamaChatMessage
@end

@implementation LlamaModel {
    llama_model *model;
}

- (instancetype)init:(llama_model *)l_model {
    self = [super init];
    if (self) {
        model = l_model;
    }
    return self;
}

- (void)dealloc
{
    llama_free_model(model);
    [super dealloc];
}

- (BOOL)addBOSToken {
    return llama_add_bos_token(model);
}

- (BOOL)addEOSToken {
    return llama_add_eos_token(model);
}

- (LlamaToken)tokenBOS {
    return llama_token_bos(model);
}

- (int32_t)nCtxTrain {
    return llama_n_ctx_train(model);
}

- (NSString *)formatExample:(NSString *)tmpl {
    return [[NSString alloc] initWithCString:common_chat_format_example(model, [tmpl cStringUsingEncoding:NSUTF8StringEncoding]).c_str()
                                    encoding:NSUTF8StringEncoding];
}

- (BOOL)hasEncoder {
    return llama_model_has_encoder(model);
}

- (llama_model *)cModel {
    return model;
}

- (BOOL)tokenIsEOG:(LlamaToken)token {
    return llama_token_is_eog(model, token);
}

- (LlamaToken)tokenEOT {
    return llama_token_eot(model);
}

- (LlamaToken)tokenEOS {
    return llama_token_eos(model);
}

@end
