#ifndef LlamaSession_Private_hpp
#define LlamaSession_Private_hpp

#import "LlamaSession.h"

@class GPTSampler;

@interface LlamaSession()

@property (atomic, strong) NSMutableString *mutableLastOutput;
@property (nonatomic, strong) GPTParams *params;
@property (nonatomic, strong) GPTSampler *smpl;

@end

#endif /* LlamaSession_Private_hpp */
