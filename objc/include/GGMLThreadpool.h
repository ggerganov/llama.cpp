#ifndef GGMLThreadpool_h
#define GGMLThreadpool_h

typedef NS_ENUM(NSUInteger, GGMLSchedPriority) {
    GGMLSchedPriorityNormal = 0,  // Normal priority
    GGMLSchedPriorityMedium = 1,  // Medium priority
    GGMLSchedPriorityHigh = 2,    // High priority
    GGMLSchedPriorityRealtime = 3 // Realtime priority
};


@interface GGMLThreadpool : NSObject
@end

@interface GGMLThreadpoolParams : NSObject

@property (nonatomic, assign) int nThreads;
@property (nonatomic, assign) GGMLSchedPriority priority;
@property (nonatomic, assign) uint32_t poll;
@property (nonatomic, assign) BOOL strictCPU;
@property (nonatomic, assign) BOOL paused;

- (BOOL)getCpuMaskAtIndex:(NSUInteger)index;
- (void)setCpuMask:(BOOL)value atIndex:(NSUInteger)index;
- (GGMLThreadpool *)threadpool;

@end

#endif /* GGMLThreadpool_h */
