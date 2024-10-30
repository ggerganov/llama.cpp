#ifndef LlamaSession_h
#define LlamaSession_h

@class GPTParams;
@class LlamaModel;
@class LlamaContext;

@interface BlockingLineQueue : NSObject

- (void)addInputLine:(NSString *)line;
- (NSString *)inputLine;
- (void)addOutputLine:(NSString *)line;
- (NSString *)outputLine;

@end

@interface LlamaSession : NSObject

@property (nonatomic, strong) LlamaModel *model;
@property (nonatomic, strong) LlamaContext *ctx;

- (instancetype)initWithParams:(GPTParams *)params;
- (void)start:(BlockingLineQueue *)queue;

@end

#endif /* Header_h */
