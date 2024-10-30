#ifndef LlamaModel_Private_hpp
#define LlamaModel_Private_hpp

#import "LlamaModel.h"
#import "llama.h"

@interface LlamaModel()

- (instancetype)init:(llama_model *)model;

- (llama_model *)cModel;

@end

#endif /* LlamaModel_Private_hpp */
