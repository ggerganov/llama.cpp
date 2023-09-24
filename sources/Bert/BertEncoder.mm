//
//  BertEncoder.mm
//  Bert
//
//  Created by Marc Terns on 9/10/23.
//

#include "BertEncoder.h"
#include "bert.h"
#include <vector>
#include <string>
#include <thread>
#import <Foundation/Foundation.h>
#import <NaturalLanguage/NaturalLanguage.h>
#import "NSArray+Vector.h"

@interface BertEncoder ()
@property (nonatomic, assign, nullable) struct bert_ctx *bctx;
@property (nonatomic, assign) int n_embd;
@property (nonatomic, strong) NSURL *modelURL;
@property (nonatomic, assign) int n_threads;
@end

@implementation BertEncoder
- (instancetype)initWithModelURL:(NSURL *)modelURL {
    if (self = [super init]) {
        self.modelURL = modelURL;
        unsigned int threads = std::thread::hardware_concurrency();
        self.n_threads = threads > 0 ? (threads <= 4 ? threads : threads / 2) : 4;
    }
    return self;
}

- (void)start {
    self.bctx = bert_load_from_file(self.modelURL.path.UTF8String);
    self.n_embd = bert_n_embd(self.bctx);
}

- (void)stop {
    bert_free(self.bctx);
}

- (std::vector<std::vector<float>>)embeddingsFromResourceURL:(NSURL *)resourceURL {
    NSMutableArray *texts = [self sentencesFromResourceURL:resourceURL];
    
    std::vector<std::vector<float>> allEmbeddings;
    for (NSString *text in texts) {
        std::vector<float> embeddings(self.n_embd);
        const char *input_str = [text UTF8String];
        bert_encode(self.bctx, self.n_threads, input_str, embeddings.data());
        allEmbeddings.push_back(embeddings);
    }
    return allEmbeddings;
}

- (std::vector<float>)embeddingsForSentence:(NSString *)sentence {
    std::vector<float> inputEmbedding = std::vector<float>(self.n_embd);
    bert_encode(self.bctx, self.n_threads, sentence.UTF8String, inputEmbedding.data());
    return inputEmbedding;
}

// Function to find the N most similar texts
-(std::vector<std::pair<float, size_t>>)findTopNSimilarInputVector:(const std::vector<float>&)inputVector textVectors:(const std::vector<std::vector<float>>&)textVectors N:(size_t)N {
    std::vector<std::pair<float, size_t>> similarities;

    for (size_t i = 0; i < textVectors.size(); ++i) {
        float similarity = cosineSimilarity(inputVector, textVectors[i]);
        similarities.emplace_back(similarity, i);
    }

    // Sort the similarities in descending order
    std::sort(similarities.begin(), similarities.end(), std::greater<std::pair<float, size_t>>());

    // Get the top N similar texts
    std::vector<std::pair<float, size_t>> topNSimilarities(similarities.begin(), similarities.begin() + N);

    return topNSimilarities;
}

- (std::vector<float>)calculateMean:(const std::vector<std::vector<float>>&)sentenceEmbeddings {
    size_t numSentences = sentenceEmbeddings.size();
    if (numSentences == 0) {
        // We might want to handle this case at the application level.
        return std::vector<float>();
    }

    // Determine the maximum embedding size among all sentence embeddings.
    // We will handle embeddings with different dimensions.
    // Alternitevly, we could check if the size of the sentence embedding
    // matches the expected size.
    size_t maxEmbeddingSize = 0;
    for (const auto& sentenceEmbedding : sentenceEmbeddings) {
        maxEmbeddingSize = std::max(maxEmbeddingSize, sentenceEmbedding.size());
    }

    std::vector<float> documentEmbedding(maxEmbeddingSize, 0.0);
    for (const auto& sentenceEmbedding : sentenceEmbeddings) {
        for (size_t i = 0; i < sentenceEmbedding.size(); ++i) {
            documentEmbedding[i] += sentenceEmbedding[i];
        }
    }

    // Calculate the mean by dividing by the number of sentences
    for (size_t i = 0; i < maxEmbeddingSize; ++i) {
        documentEmbedding[i] /= numSentences;
    }

    return documentEmbedding;
}

- (NSArray<NSString *> *)findClosestTextForSentence:(NSString *)sentence inResourceURL:(NSURL *)resourceURL topN:(NSInteger)topN {
    std::vector<std::vector<float>> allEmbeddings = [self embeddingsFromResourceURL:resourceURL];
    std::vector<float> sentenceEmbedding = [self embeddingsForSentence:sentence];
    std::vector<std::pair<float, size_t>> topNSimilarities = [self findTopNSimilarInputVector:sentenceEmbedding textVectors:allEmbeddings N:topN];
    NSMutableArray *sentenceArray = [self sentencesFromResourceURL:resourceURL];

    NSMutableArray<NSString *> *result = [NSMutableArray new];
    for (const auto& similarity : topNSimilarities) {
        size_t index = similarity.second;
        // Check if the index is within the bounds of the 'texts' array
        if (index < [sentenceArray count]) {
            NSString *similarText = sentenceArray[index];
            [result addObject:similarText];
        }
    }
    return [result copy];
}

- (NSArray<NSString *> *)sentencesFromResourceURL:(NSURL *)resourceURL {
    NSError *error;
    NSString *fileContents = [NSString stringWithContentsOfFile:resourceURL.path encoding:NSUTF8StringEncoding error:&error];
    
    if (error) {
        NSLog(@"Error reading file: %@", error.localizedDescription);
        return @[];
    }

    NLTokenizer *tokenizer = [[NLTokenizer alloc] initWithUnit:NLTokenUnitSentence];
    [tokenizer setString:fileContents];
    
    NSMutableArray *sentenceArray = [NSMutableArray array];
    
    [tokenizer enumerateTokensInRange:NSMakeRange(0, [fileContents length])
                               usingBlock:^(NSRange tokenRange, NLTokenizerAttributes attributes, BOOL *stop) {
        NSString *sentence = [fileContents substringWithRange:tokenRange];
        // Check if the sentence is not empty or only consists of whitespace
        if ([[sentence stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] length] > 0) {
            [sentenceArray addObject:sentence];
        }
    }];
    return [sentenceArray copy];
}

-(std::vector<std::string>)selectSentences:(const std::vector<std::vector<float>>&)sentenceEmbeddings documentEmbedding:(const std::vector<float>&)documentEmbedding threshold:(double)threshold sentences:(NSMutableArray<NSString *> *)sentences {
    std::vector<std::string> selectedSentences;

    for (size_t i = 0; i < sentenceEmbeddings.size(); ++i) {
        double similarity = cosineSimilarity(sentenceEmbeddings[i], documentEmbedding);
        // Check if the similarity score is above the threshold.
        // Small modifications on threshold can have a huge impact on similarity.
        // 0.7 vs 0.77 can make a big difference in sumarization.
        if (similarity >= threshold) {
            if (i < [sentences count]) {
                // We extract the actual text sentence from the original document
                // given its score.
                selectedSentences.push_back([sentences objectAtIndex:i].UTF8String);
            }
        }
    }

    return selectedSentences;
}

- (NSString *)summarizeFromResourceURL:(NSURL *)resourceURL threshold:(double)threshold {
    std::vector<std::vector<float>> sentenceEmbeddings = [self embeddingsFromResourceURL:resourceURL];
    std::vector<float> documentEmbedding = [self calculateMean:sentenceEmbeddings];
    NSMutableArray *sentences = [self sentencesFromResourceURL:resourceURL];

    // Select sentences based on the threshold
    std::vector<std::string> selectedSentences = [self selectSentences:sentenceEmbeddings documentEmbedding:documentEmbedding threshold:threshold sentences:sentences];
    std::string summary;

    for (const auto& sentence : selectedSentences) {
        summary += sentence;
    }

    return [NSString stringWithUTF8String:summary.c_str()];
}

- (BERTEmbeddingsData *)embeddingsForResourceURL:(NSURL *)resourceURL {
    std::vector<std::vector<float>> allEmbeddings = [self embeddingsFromResourceURL:resourceURL];
    NSMutableArray<NSArray<NSNumber *> *> *result = [NSMutableArray array];
    for (const auto& innerVector : allEmbeddings) {
        NSMutableArray<NSNumber *> *innerArray = [NSMutableArray array];
        for (float floatValue : innerVector) {
            [innerArray addObject:@(floatValue)];
        }
        [result addObject:innerArray];
    }
    BERTEmbeddingsData *data = [[BERTEmbeddingsData alloc] initWithResourceURL:resourceURL embeddings:result];
    return data;
}

- (NSArray<NSString *> *)findClosestTextForSentence:(NSString *)sentence embeddingsData:(NSArray<BERTEmbeddingsData *> *)embeddingsData topN:(NSInteger)topN {
    NSMutableArray<NSString *> *closestTexts = [NSMutableArray array];

    std::vector<float> inputEmbedding = [self embeddingsForSentence:sentence];
    std::vector<std::pair<float, NSString *>> allSimilarities;
    
    for (BERTEmbeddingsData *data in embeddingsData) {
        // Original sentences array from the resource file. we will use this to find the sentence match after applying the math on the embeddings.
        NSMutableArray<NSString *> *sentenceArray = [self sentencesFromResourceURL:[data resourceURL]];
        
        std::vector<std::vector<float>> allEmbeddings = [[data embeddings] stdVectorArray];
        std::vector<std::pair<float, size_t>> topNSimilarities = [self findTopNSimilarInputVector:inputEmbedding textVectors:allEmbeddings N:topN];
        
        // Get the actual sentences corresponding to the indices and store them with their similarity scores
        for (const auto& similarity : topNSimilarities) {
            size_t index = similarity.second;
            if (index < [sentenceArray count]) {
                NSString *similarText = sentenceArray[index];
                float similarityScore = similarity.first;
                // Storing the tuple will allow us to later sort based on similarity score.
                allSimilarities.push_back(std::make_pair(similarityScore, similarText));
            }
        }
    }
    
    // We need to sort based on similarity score to make sure we get the most accurate results across all files.
    std::sort(allSimilarities.begin(), allSimilarities.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Return the top N closest texts from all data combined
    for (size_t i = 0; i < MIN(topN, allSimilarities.size()); ++i) {
        [closestTexts addObject:allSimilarities[i].second];
    }
    
    return [closestTexts copy];
}

// Calculate cosine similarity between two vectors
float cosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    // Check if the vectors have the same length
    if (vec1.size() != vec2.size()) {
        return 0.0;
    }

    // Calculate dot product
    float dotProduct = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
    }

    // Calculate magnitudes
    float magnitude1 = 0.0;
    float magnitude2 = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        magnitude1 += vec1[i] * vec1[i];
        magnitude2 += vec2[i] * vec2[i];
    }

    // Calculate cosine similarity
    if (magnitude1 == 0.0 || magnitude2 == 0.0) {
        return 0.0;
    } else {
        return dotProduct / (sqrt(magnitude1) * sqrt(magnitude2));
    }
}

@end
