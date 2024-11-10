#ifndef LlamaModel_h
#define LlamaModel_h

@class GPTParams;
@class GGMLThreadpool;
@class LlamaContext;

typedef int32_t LlamaToken;

@interface LlamaChatMessage : NSObject

@property (nonatomic, copy) NSString *role;
@property (nonatomic, copy) NSString *content;

@end

@interface LlamaContextParams : NSObject
@end

@interface LlamaModel : NSObject

- (LlamaToken)tokenBOS;
- (LlamaToken)tokenEOT;
- (LlamaToken)tokenEOS;
- (BOOL)tokenIsEOG:(LlamaToken)token;
- (int32_t)nCtxTrain;
- (BOOL)addBOSToken;
- (BOOL)addEOSToken;
- (BOOL)hasEncoder;
- (NSString *)formatExample:(NSString *)tmpl;

@end

#endif /* LlamaModel_h */
