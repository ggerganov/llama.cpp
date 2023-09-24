//
//  EmbeddingsData.h
//  
//
//  Created by Marc Terns on 9/23/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface BERTEmbeddingsData : NSObject

@property (nonatomic, strong) NSURL *resourceURL;
@property (nonatomic, strong) NSArray<NSArray<NSNumber *> *> *embeddings;

- (instancetype)initWithResourceURL:(NSURL *)resourceURL embeddings:(NSArray<NSArray<NSNumber *> *> *)embeddings;

@end


NS_ASSUME_NONNULL_END
