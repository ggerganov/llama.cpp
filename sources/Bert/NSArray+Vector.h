//
//  NSArray+Vector.h
//  
//
//  Created by Marc Terns on 9/23/23.
//

#import <Foundation/Foundation.h>
#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface NSArray (Vector)

- (std::vector<std::vector<float>>)stdVectorArray;

@end

NS_ASSUME_NONNULL_END
