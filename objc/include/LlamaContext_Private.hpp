#ifndef LlamaContext_Private_hpp
#define LlamaContext_Private_hpp

#import "LlamaContext.h"
#import "../../common/common.h"

@interface LlamaContext()

- (instancetype)initWithContext:(llama_context *)context;

- (std::vector<llama_token>)tokenize:(NSString *)text
                          addSpecial:(BOOL)addSpecial
                        parseSpecial:(BOOL)parseSpecial;

- (BOOL)loadStateFile:(NSString *)pathSession
            tokensOut:(llama_token *)tokensOut
        nTokenCpacity:(size_t)nTokenCapacity
       nTokenCountOut:(size_t *)nTokenCountOut;

- (std::string)convertTokensToString:(const std::vector<llama_token>&)tokens;

- (llama_context *)cContext;

- (int32_t)encode:(llama_batch)batch;

@end

#endif /* LlamaContext_Private_hpp */
