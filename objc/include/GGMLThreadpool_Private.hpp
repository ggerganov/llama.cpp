#ifndef GGMLThreadpool_Private_hpp
#define GGMLThreadpool_Private_hpp

#import "GGMLThreadpool.h"
#import "../../common/common.h"

@interface GGMLThreadpool() {
    ggml_threadpool *threadpool;
}

- (instancetype)initWithThreadpool:(ggml_threadpool *)threadpool;
- (ggml_threadpool *)threadpool;

@end

@interface GGMLThreadpoolParams()

- (instancetype)initWithParams:(ggml_threadpool_params&&)params;

@end

#endif /* GGMLThreadpool_Private_hpp */
