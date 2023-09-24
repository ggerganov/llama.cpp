//
//  NSArray+Vector.m
//  
//
//  Created by Marc Terns on 9/23/23.
//

#import "NSArray+Vector.h"

@implementation NSArray (Vector)

- (std::vector<std::vector<float>>)stdVectorArray {
    std::vector<std::vector<float>> result;
    
    for (NSArray<NSNumber *> *innerArray in self) {
        std::vector<float> innerVector;
        
        for (NSNumber *number in innerArray) {
            float floatValue = [number floatValue];
            innerVector.push_back(floatValue);
        }
        
        result.push_back(innerVector);
    }
    
    return result;
}

@end
