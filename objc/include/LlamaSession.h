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

@property (nonatomic, assign) BOOL isClosed;

@end

NS_REFINED_FOR_SWIFT @interface LlamaSession : NSObject

@property (nonatomic, strong) LlamaModel *model;
@property (nonatomic, strong) LlamaContext *ctx;
@property (nonatomic, strong, readonly) NSString *lastOutput;
@property (nonatomic, strong) BlockingLineQueue *queue;

- (instancetype)initWithParams:(GPTParams *)params;
- (void)start;
- (void)stop;

@end

#endif /* Header_h */
