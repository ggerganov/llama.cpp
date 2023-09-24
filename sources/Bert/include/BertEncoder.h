//
//  BertEncoder.h
//  Bert
//
//  Created by Marc Terns on 9/10/23.
//

#import <Foundation/Foundation.h>
#import "BERTEmbeddingsData.h"

NS_ASSUME_NONNULL_BEGIN

@interface BertEncoder : NSObject

- (instancetype)initWithModelURL:(NSURL *)modelURL;

- (void)start;

- (void)stop;

/// Given a sentence and fa file to to look from,  returns the top N results that closely match the input sentence in semantic meaning.
///
/// - Parameters:
///   - sentence: The input sentence you want to find the closest texts from.
///   - resouceURL: The file you want to find similarities from.
///   - topN: The topN results.
///
- (NSArray<NSString *> *)findClosestTextForSentence:(NSString *)sentence inResourceURL:(NSURL *)resourceURL topN:(NSInteger)topN;

/// Given a text file, and a threshold, summarizes the contents within the resource file.
/// - Parameters:
///   - resourceURL: The file to summarize
///   - threshold: The threhsold for summarization. Values range from 0.0 to 1.0. The higher the value, the smaller the length of the summary.
///                Values over 0.79 might not produce a summary.
///                For consice summary, with only the most critical information, you might set a relatively high threshold (0.7 or higher).
///                The larger the text, the larget the customization of threshold values.
///
- (NSString *)summarizeFromResourceURL:(NSURL *)resourceURL threshold:(double)threshold;

/// Returns all the sentence embeddings found in a given file. This API allows customers to store their own embeddings to disk and avoid having to calculate them every tine.,
/// - Parameters:
///   - resourceURL: The file where you want to get the embeddings from.
///   
- (BERTEmbeddingsData *)embeddingsForResourceURL:(NSURL *)resourceURL;

/// Given a sentence and embeddings to look from,  returns the top N results that closely match the input sentence in semantic meaning.
///
/// - Parameters:
///   - sentence: The input sentence you want to find the closest texts from.
///   - embeddingsData: The embeddings to perform the similarity check on the sentence.
///   - topN: The topN results.
///
- (NSArray<NSString *> *)findClosestTextForSentence:(NSString *)sentence embeddingsData:(NSArray<BERTEmbeddingsData *> *)embeddingsData topN:(NSInteger)topN;

@end

NS_ASSUME_NONNULL_END
