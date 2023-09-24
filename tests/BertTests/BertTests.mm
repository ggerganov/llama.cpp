//
//  BertTests.m
//  
//
//  Created by Marc Terns on 9/13/23.
//

#import <XCTest/XCTest.h>
#import "BertEncoder.h"

@interface BertTests : XCTestCase

@end

@implementation BertTests

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testSimilarTexts {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"earnings" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    NSArray<NSString *> * result = [encoder findClosestTextForSentence:@"Who is the CEO?" inResourceURL:earningsResourceURL topN:3];
    NSLog(@"%@", result);
    XCTAssertEqual(result.count, 3);
    XCTAssertTrue([result.firstObject isEqualToString:@"Jay Chaudhry -- Founder, Chairman, and Chief Executive Officer\n"]);
}

- (void)testSimilarTextsFromEmbeddings {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"earnings" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    BERTEmbeddingsData *data = [encoder embeddingsForResourceURL:earningsResourceURL];
    NSArray<NSString *> * result = [encoder findClosestTextForSentence:@"Who is the CEO?" embeddingsData:@[data] topN:3];
    NSLog(@"%@", result);
    XCTAssertEqual(result.count, 3);
    XCTAssertTrue([result.firstObject isEqualToString:@"Jay Chaudhry -- Founder, Chairman, and Chief Executive Officer\n"]);
}

- (void)testSimilarTextsFromEmbeddingsInMultipleFiles {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"earnings" withExtension:@"txt"];
    NSURL *example_EN = [bundle URLForResource:@"example_EN" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    BERTEmbeddingsData *earningsData = [encoder embeddingsForResourceURL:earningsResourceURL];
    BERTEmbeddingsData *exampleData = [encoder embeddingsForResourceURL:example_EN];
    NSArray<NSString *> * result = [encoder findClosestTextForSentence:@"What is summarization?" embeddingsData:@[earningsData, exampleData] topN:3];
    NSLog(@"%@", result);
    XCTAssertEqual(result.count, 3);
    XCTAssertTrue([result.firstObject isEqualToString:@"In text summarization, the goal is to create a concise and coherent summary of a document. "]);
}

- (void)testSummary_earnings {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"earnings" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    NSLog(@"%@", [encoder summarizeFromResourceURL:earningsResourceURL threshold:0.50]);
}

- (void)testSummary_EN {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"example_EN" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    NSLog(@"%@", [encoder summarizeFromResourceURL:earningsResourceURL threshold:0.75]);
}

- (void)testSummary_large_EN {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"example_large_EN" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    NSLog(@"%@", [encoder summarizeFromResourceURL:earningsResourceURL threshold:0.6]);
}

-(void)testSummary_ES {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"example_ES" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    NSLog(@"%@", [encoder summarizeFromResourceURL:earningsResourceURL threshold:0.75]);
}

-(void)testSummary_CAT {
    NSBundle *bundle = SWIFTPM_MODULE_BUNDLE;
    NSURL *resourceURL = [bundle URLForResource:@"ggml-model-f32" withExtension:@"bin"];
    NSURL *earningsResourceURL = [bundle URLForResource:@"example_CAT" withExtension:@"txt"];
    BertEncoder * encoder = [[BertEncoder alloc] initWithModelURL:resourceURL];
    [encoder start];
    NSLog(@"%@", [encoder summarizeFromResourceURL:earningsResourceURL threshold:0.55]);
}
@end
