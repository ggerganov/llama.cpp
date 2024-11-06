#ifndef CPUParams_Private_hpp
#define CPUParams_Private_hpp

#import "CPUParams.h"
#import "../../common/common.h"

@interface CPUParams() {
    cpu_params *params;
}

- (instancetype)initWithParams:(cpu_params&)params;

@end

#endif /* CPUParams_Private_hpp */
