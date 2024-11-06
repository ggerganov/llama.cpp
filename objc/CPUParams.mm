#import <Foundation/Foundation.h>
#import "CPUParams_Private.hpp"
#import "GGMLThreadpool_Private.hpp"

@implementation CPUParams

- (instancetype)initWithParams:(cpu_params&)params
{
    self = [super init];
    if (self) {
        self->params = &params;
    }
    return self;
}

- (NSInteger)nThreads {
    return params->n_threads;
}

- (void)setNThreads:(NSInteger)nThreads {
    params->n_threads = nThreads;
}

- (BOOL)maskValid {
    return params->mask_valid;
}

- (void)setMaskValid:(BOOL)maskValid {
    params->mask_valid = maskValid;
}

- (GGMLSchedPriority)priority {
    return GGMLSchedPriority(params->priority);
}

- (void)setPriority:(GGMLSchedPriority)priority {
    params->priority = ggml_sched_priority(priority);
}

- (BOOL)strictCPU {
    return params->strict_cpu;
}

- (void)setStrictCPU:(BOOL)strictCPU {
    params->strict_cpu = strictCPU;
}

- (NSUInteger)poll {
    return params->poll;
}

- (void)setPoll:(NSUInteger)poll {
    params->poll = poll;
}

- (BOOL)getCpuMaskAtIndex:(NSUInteger)index {
    return params->cpumask[index];
}

- (void)setCpuMask:(BOOL)value atIndex:(NSUInteger)index {
    params->cpumask[index] = value;
}

- (GGMLThreadpoolParams *)ggmlThreadpoolParams {
    return [[GGMLThreadpoolParams alloc] initWithParams:ggml_threadpool_params_from_cpu_params(*params)];
}

@end
