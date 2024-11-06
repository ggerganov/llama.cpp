#ifndef CPUParams_h
#define CPUParams_h

typedef NS_ENUM(NSUInteger, GGMLSchedPriority);

@class GGMLThreadpoolParams;

@interface CPUParams : NSObject

/// Number of threads to use
@property (nonatomic, assign) NSInteger nThreads;

/// Default: any CPU
@property (nonatomic, assign) BOOL maskValid;
/// Scheduling priority
@property (nonatomic, assign) GGMLSchedPriority priority;
/// Use strict CPU placement
@property (nonatomic, assign) BOOL strictCPU;
/// Polling (busywait) level (0 - no polling, 100 - mostly polling)
@property (nonatomic, assign) NSUInteger poll;

// Custom methods to access or manipulate the cpumask array
- (BOOL)getCpuMaskAtIndex:(NSUInteger)index;
- (void)setCpuMask:(BOOL)value atIndex:(NSUInteger)index;

- (GGMLThreadpoolParams *)ggmlThreadpoolParams;

@end

#endif /* Header_h */
