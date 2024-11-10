#import <Foundation/Foundation.h>
#import "GGMLThreadpool_Private.hpp"

@implementation GGMLThreadpool

- (instancetype)initWithThreadpool:(ggml_threadpool *)threadpool
{
    self = [super init];
    if (self) {
        self->threadpool = threadpool;
    }
    return self;
}

- (ggml_threadpool *)threadpool {
    return self->threadpool;
}

- (void)dealloc
{
    ggml_threadpool_free(threadpool);
    [super dealloc];
}
@end

@implementation GGMLThreadpoolParams {
    ggml_threadpool_params params;
}

- (BOOL)getCpuMaskAtIndex:(NSUInteger)index {
    abort();
}

- (void)setCpuMask:(BOOL)value atIndex:(NSUInteger)index {
    abort();
}

- (instancetype)initWithParams:(ggml_threadpool_params&&)params
{
    self = [super init];
    if (self) {
        self->params = params;
    }
    return self;
}

- (BOOL)isEqual:(id)other {
    GGMLThreadpoolParams *rhs = (GGMLThreadpoolParams *)other;
    ggml_threadpool_params rhs_params = rhs->params;
    return ggml_threadpool_params_match(&params, &rhs_params);
}

- (GGMLThreadpool *)threadpool {
    auto tp = ggml_threadpool_new(&params);
    return [[GGMLThreadpool alloc] initWithThreadpool:tp];
}
@end
