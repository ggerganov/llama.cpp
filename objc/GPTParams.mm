#import <Foundation/Foundation.h>
#import "GPTParams_Private.hpp"
#import "CPUParams_Private.hpp"
#import "GPTSampler.h"
#import "../common/common.h"
#import "ggml.h"

@implementation GPTSamplerParams {
    common_sampler_params *gpt_sampler_params;
}

- (instancetype)initWithParams:(common_sampler_params&)params {
    self = [super init];
    if (self) {
        gpt_sampler_params = &params;
    }
    return self;
}

// Getters and setters for Objective-C properties, which manipulate the C++ struct

- (uint32_t)seed {
    return gpt_sampler_params->seed;
}

- (void)setSeed:(uint32_t)seed {
    gpt_sampler_params->seed = seed;
}

- (int32_t)nPrev {
    return gpt_sampler_params->n_prev;
}

- (void)setNPrev:(int32_t)nPrev {
    gpt_sampler_params->n_prev = nPrev;
}

- (int32_t)nProbs {
    return gpt_sampler_params->n_probs;
}

- (void)setNProbs:(int32_t)nProbs {
    gpt_sampler_params->n_probs = nProbs;
}

- (int32_t)minKeep {
    return gpt_sampler_params->min_keep;
}

- (void)setMinKeep:(int32_t)minKeep {
    gpt_sampler_params->min_keep = minKeep;
}

- (int32_t)topK {
    return gpt_sampler_params->top_k;
}

- (void)setTopK:(int32_t)topK {
    gpt_sampler_params->top_k = topK;
}

- (float)topP {
    return gpt_sampler_params->top_p;
}

- (void)setTopP:(float)topP {
    gpt_sampler_params->top_p = topP;
}

- (float)minP {
    return gpt_sampler_params->min_p;
}

- (void)setMinP:(float)minP {
    gpt_sampler_params->min_p = minP;
}

//- (float)tfsZ {
//    return gpt_sampler_params->tfs_z;
//}
//
//- (void)setTfsZ:(float)tfsZ {
//    gpt_sampler_params->tfs_z = tfsZ;
//}

- (float)typP {
    return gpt_sampler_params->typ_p;
}

- (void)setTypP:(float)typP {
    gpt_sampler_params->typ_p = typP;
}

- (float)temp {
    return gpt_sampler_params->temp;
}

- (void)setTemp:(float)temp {
    gpt_sampler_params->temp = temp;
}

- (float)dynatempRange {
    return gpt_sampler_params->dynatemp_range;
}

- (void)setDynatempRange:(float)dynatempRange {
    gpt_sampler_params->dynatemp_range = dynatempRange;
}

- (float)dynatempExponent {
    return gpt_sampler_params->dynatemp_exponent;
}

- (void)setDynatempExponent:(float)dynatempExponent {
    gpt_sampler_params->dynatemp_exponent = dynatempExponent;
}

- (int32_t)penaltyLastN {
    return gpt_sampler_params->penalty_last_n;
}

- (void)setPenaltyLastN:(int32_t)penaltyLastN {
    gpt_sampler_params->penalty_last_n = penaltyLastN;
}

- (float)penaltyRepeat {
    return gpt_sampler_params->penalty_repeat;
}

- (void)setPenaltyRepeat:(float)penaltyRepeat {
    gpt_sampler_params->penalty_repeat = penaltyRepeat;
}

- (float)penaltyFreq {
    return gpt_sampler_params->penalty_freq;
}

- (void)setPenaltyFreq:(float)penaltyFreq {
    gpt_sampler_params->penalty_freq = penaltyFreq;
}

- (float)penaltyPresent {
    return gpt_sampler_params->penalty_present;
}

- (void)setPenaltyPresent:(float)penaltyPresent {
    gpt_sampler_params->penalty_present = penaltyPresent;
}

- (int32_t)mirostat {
    return gpt_sampler_params->mirostat;
}

- (void)setMirostat:(int32_t)mirostat {
    gpt_sampler_params->mirostat = mirostat;
}

- (float)mirostatTau {
    return gpt_sampler_params->mirostat_tau;
}

- (void)setMirostatTau:(float)mirostatTau {
    gpt_sampler_params->mirostat_tau = mirostatTau;
}

- (float)mirostatEta {
    return gpt_sampler_params->mirostat_eta;
}

- (void)setMirostatEta:(float)mirostatEta {
    gpt_sampler_params->mirostat_eta = mirostatEta;
}

- (BOOL)penalizeNl {
    return gpt_sampler_params->penalize_nl;
}

- (void)setPenalizeNl:(BOOL)penalizeNl {
    gpt_sampler_params->penalize_nl = penalizeNl;
}

- (BOOL)ignoreEos {
    return gpt_sampler_params->ignore_eos;
}

- (void)setIgnoreEos:(BOOL)ignoreEos {
    gpt_sampler_params->ignore_eos = ignoreEos;
}

- (BOOL)noPerf {
    return gpt_sampler_params->no_perf;
}

- (void)setNoPerf:(BOOL)noPerf {
    gpt_sampler_params->no_perf = noPerf;
}

// For `samplers`, convert from NSArray<NSNumber *> to std::vector
- (NSArray<NSNumber *> *)samplers {
    NSMutableArray<NSNumber *> *samplersArray = [NSMutableArray array];
    for (auto sampler : gpt_sampler_params->samplers) {
        [samplersArray addObject:@(sampler)];
    }
    return [samplersArray copy];
}

- (void)setSamplers:(NSArray<NSNumber *> *)samplers {
    gpt_sampler_params->samplers.clear();
    for (NSNumber *sampler in samplers) {
        gpt_sampler_params->samplers.push_back(static_cast<common_sampler_type>(sampler.intValue));
    }
}

//// For `logitBias`, convert from NSArray<NSNumber *> to std::vector
//- (NSArray<NSNumber *> *)logitBias {
//    NSMutableArray<llama_logit_bias *> *logitBiasArray = [NSMutableArray array];
//    for (auto bias : gpt_sampler_params.logit_bias) {
//        [logitBiasArray addObject:bias];
//    }
//    return [logitBiasArray copy];
//}
//
//- (void)setLogitBias:(NSArray<NSNumber *> *)logitBias {
//    gpt_sampler_params.logit_bias.clear();
//    for (NSNumber *bias in logitBias) {
//        gpt_sampler_params.logit_bias.push_back(bias.floatValue);
//    }
//}

// For `grammar`, convert between NSString and std::string
- (NSString *)grammar {
    return [NSString stringWithUTF8String:gpt_sampler_params->grammar.c_str()];
}

- (void)setGrammar:(NSString *)grammar {
    gpt_sampler_params->grammar = std::string([grammar UTF8String]);
}

// Method to print out the parameters as a string
- (NSString *)print {
    NSMutableString *output = [NSMutableString stringWithString:@"GPT Sampler Params:\n"];
    [output appendFormat:@"Seed: %u\n", self.seed];
    [output appendFormat:@"nPrev: %d\n", self.nPrev];
    [output appendFormat:@"nProbs: %d\n", self.nProbs];
    [output appendFormat:@"minKeep: %d\n", self.minKeep];
    [output appendFormat:@"topK: %d\n", self.topK];
    [output appendFormat:@"topP: %.2f\n", self.topP];
    [output appendFormat:@"minP: %.2f\n", self.minP];
    [output appendFormat:@"tfsZ: %.2f\n", self.tfsZ];
    [output appendFormat:@"typP: %.2f\n", self.typP];
    [output appendFormat:@"temp: %.2f\n", self.temp];
    [output appendFormat:@"dynatempRange: %.2f\n", self.dynatempRange];
    [output appendFormat:@"dynatempExponent: %.2f\n", self.dynatempExponent];
    [output appendFormat:@"penaltyLastN: %d\n", self.penaltyLastN];
    [output appendFormat:@"penaltyRepeat: %.2f\n", self.penaltyRepeat];
    [output appendFormat:@"penaltyFreq: %.2f\n", self.penaltyFreq];
    [output appendFormat:@"penaltyPresent: %.2f\n", self.penaltyPresent];
    [output appendFormat:@"mirostat: %d\n", self.mirostat];
    [output appendFormat:@"mirostatTau: %.2f\n", self.mirostatTau];
    [output appendFormat:@"mirostatEta: %.2f\n", self.mirostatEta];
    [output appendFormat:@"penalizeNl: %@\n", self.penalizeNl ? @"YES" : @"NO"];
    [output appendFormat:@"ignoreEos: %@\n", self.ignoreEos ? @"YES" : @"NO"];
    [output appendFormat:@"noPerf: %@\n", self.noPerf ? @"YES" : @"NO"];
    [output appendFormat:@"Grammar: %@\n", self.grammar];
    
    // Print samplers
    [output appendString:@"Samplers: "];
    for (NSNumber *sampler in self.samplers) {
        [output appendFormat:@"%d, ", sampler.intValue];
    }
    [output appendString:@"\n"];
    
    // Print logit biases
    [output appendString:@"Logit Biases: "];
    for (NSNumber *bias in self.logitBias) {
        [output appendFormat:@"%.2f, ", bias.floatValue];
    }
    [output appendString:@"\n"];

    return [output copy];
}

- (common_sampler_params&)cParams {
    return *gpt_sampler_params;
}

@end

@implementation GPTParams {
    common_params gpt_params;
}

- (NSArray<NSString *> *)antiPrompts {
    auto antiprompts = [[NSMutableArray alloc] init];
    for (auto& antiprompt : gpt_params.antiprompt) {
        [antiprompts addObject:[NSString stringWithCString:antiprompt.c_str() encoding:NSUTF8StringEncoding]];
    }
    return antiprompts;
}

- (void)setAntiPrompts:(NSArray<NSString *> *)antiPrompts {
    gpt_params.antiprompt.clear();
    for (NSString *antiprompt in antiPrompts) {
        gpt_params.antiprompt.push_back([antiprompt cStringUsingEncoding:NSUTF8StringEncoding]);
    }
}

- (NSArray<NSString *> *)apiKeys {
    auto apiKeys = [[NSMutableArray alloc] init];
    for (auto& apiKey : gpt_params.api_keys) {
        [apiKeys addObject:[NSString stringWithCString:apiKey.c_str() encoding:NSUTF8StringEncoding]];
    }
    return apiKeys;
}

- (void)setApiKeys:(NSArray<NSString *> *)apiKeys {
    gpt_params.api_keys.clear();
    for (NSString *apiKey in apiKeys) {
        gpt_params.api_keys.push_back([apiKey cStringUsingEncoding:NSUTF8StringEncoding]);
    }
}

- (NSArray<NSNumber *> *)tensorSplit {
    auto tensorSplit = [[NSMutableArray alloc] init];
    for (auto& tensor : gpt_params.tensor_split) {
        [tensorSplit addObject:[[NSNumber alloc] initWithFloat:tensor]];
    }
    return tensorSplit;
}

- (void)setTensorSplit:(NSArray<NSNumber *> *)tensorSplit {
    for (size_t i = 0; i < [tensorSplit count]; i++) {
        gpt_params.tensor_split[i] = [tensorSplit[i] floatValue];
    }
}

- (common_params&)params {
    return gpt_params;
}

- (int32_t)nPredict {
    return gpt_params.n_predict;
}

- (void)setNPredict:(int32_t)nPredict {
    gpt_params.n_predict = nPredict;
}

- (NSInteger)nCtx {
    return gpt_params.n_ctx;
}

- (void)setNCtx:(NSInteger)nCtx {
    gpt_params.n_ctx = nCtx;
}

- (int32_t)nBatch {
    return gpt_params.n_batch;
}

- (void)setNBatch:(int32_t)nBatch {
    gpt_params.n_batch = nBatch;
}

- (int32_t)nUBatch {
    return gpt_params.n_ubatch;
}

- (void)setNUBatch:(int32_t)nUBatch {
    gpt_params.n_ubatch = nUBatch;
}

- (int32_t)nKeep {
    return gpt_params.n_keep;
}

- (void)setNKeep:(int32_t)nKeep {
    gpt_params.n_keep = nKeep;
}

- (int32_t)nDraft {
    return gpt_params.n_draft;
}

- (void)setNDraft:(int32_t)nDraft {
    gpt_params.n_draft = nDraft;
}

- (int32_t)nChunks {
    return gpt_params.n_chunks;
}

- (void)setNChunks:(int32_t)nChunks {
    gpt_params.n_chunks = nChunks;
}

- (int32_t)nParallel {
    return gpt_params.n_parallel;
}

- (void)setNParallel:(int32_t)nParallel {
    gpt_params.n_parallel = nParallel;
}

- (int32_t)nSequences {
    return gpt_params.n_sequences;
}

- (void)setNSequences:(int32_t)nSequences {
    gpt_params.n_sequences = nSequences;
}

- (float)pSplit {
    return gpt_params.p_split;
}

- (void)setPSplit:(float)pSplit {
    gpt_params.p_split = pSplit;
}

- (int32_t)nGpuLayers {
    return gpt_params.n_gpu_layers;
}

- (void)setNGpuLayers:(int32_t)nGpuLayers {
    gpt_params.n_gpu_layers = nGpuLayers;
}

- (int32_t)nGpuLayersDraft {
    return gpt_params.n_gpu_layers_draft;
}

- (void)setNGpuLayersDraft:(int32_t)nGpuLayersDraft {
    gpt_params.n_gpu_layers_draft = nGpuLayersDraft;
}

- (int32_t)mainGpu {
    return gpt_params.main_gpu;
}

- (void)setMainGpu:(int32_t)mainGpu {
    gpt_params.main_gpu = mainGpu;
}

- (int32_t)grpAttnN {
    return gpt_params.grp_attn_n;
}

- (void)setGrpAttnN:(int32_t)grpAttnN {
    gpt_params.grp_attn_n = grpAttnN;
}

- (int32_t)grpAttnW {
    return gpt_params.grp_attn_w;
}

- (void)setGrpAttnW:(int32_t)grpAttnW {
    gpt_params.grp_attn_w = grpAttnW;
}

- (int32_t)nPrint {
    return gpt_params.n_print;
}

- (void)setNPrint:(int32_t)nPrint {
    gpt_params.n_print = nPrint;
}

- (float)ropeFreqBase {
    return gpt_params.rope_freq_base;
}

- (void)setRopeFreqBase:(float)ropeFreqBase {
    gpt_params.rope_freq_base = ropeFreqBase;
}

- (float)ropeFreqScale {
    return gpt_params.rope_freq_scale;
}

- (void)setRopeFreqScale:(float)ropeFreqScale {
    gpt_params.rope_freq_scale = ropeFreqScale;
}

- (float)yarnExtFactor {
    return gpt_params.yarn_ext_factor;
}

- (void)setYarnExtFactor:(float)yarnExtFactor {
    gpt_params.yarn_ext_factor = yarnExtFactor;
}

- (float)yarnAttnFactor {
    return gpt_params.yarn_attn_factor;
}

- (void)setYarnAttnFactor:(float)yarnAttnFactor {
    gpt_params.yarn_attn_factor = yarnAttnFactor;
}

- (float)yarnBetaFast {
    return gpt_params.yarn_beta_fast;
}

- (void)setYarnBetaFast:(float)yarnBetaFast {
    gpt_params.yarn_beta_fast = yarnBetaFast;
}

- (float)yarnBetaSlow {
    return gpt_params.yarn_beta_slow;
}

- (void)setYarnBetaSlow:(float)yarnBetaSlow {
    gpt_params.yarn_beta_slow = yarnBetaSlow;
}

- (int32_t)yarnOrigCtx {
    return gpt_params.yarn_orig_ctx;
}

- (void)setYarnOrigCtx:(int32_t)yarnOrigCtx {
    gpt_params.yarn_orig_ctx = yarnOrigCtx;
}

- (float)defragThold {
    return gpt_params.defrag_thold;
}

- (void)setDefragThold:(float)defragThold {
    gpt_params.defrag_thold = defragThold;
}

// Assuming tensorSplit remains a fixed array in C struct, we can create a method to access specific values.
- (float)tensorSplitAtIndex:(NSUInteger)index {
    if (index < 128) {
        return gpt_params.tensor_split[index];
    }
    return 0.0f;  // Return default value if index is out of bounds
}

- (void)setTensorSplitValue:(float)value atIndex:(NSUInteger)index {
    if (index < 128) {
        gpt_params.tensor_split[index] = value;
    }
}

- (BOOL)embedding {
    return gpt_params.embedding;
}

- (void)setEmbedding:(BOOL)embedding {
    gpt_params.embedding = embedding;
}

- (LlamaModelParams *)LlamaModelParams {
    return nil;
}

- (BOOL)ctxShift {
    return gpt_params.ctx_shift;
}

- (void)setCtxShift:(BOOL)ctxShift {
    gpt_params.ctx_shift = ctxShift;
}

- (CPUParams *)cpuParams {
    return [[CPUParams alloc] initWithParams:gpt_params.cpuparams];
}

- (CPUParams *)cpuParamsBatch {
    return [[CPUParams alloc] initWithParams:gpt_params.cpuparams_batch];
}

- (GPTSamplerParams *)samplerParams {
    return [[GPTSamplerParams alloc] initWithParams:gpt_params.sparams];
}

- (NSString *)modelURL {
    return [NSString stringWithCString:gpt_params.model_url.c_str() encoding:NSUTF8StringEncoding];
}

- (void)setModelURL:(NSString *)modelURL {
    gpt_params.model_url = [modelURL cStringUsingEncoding:NSUTF8StringEncoding];
}

- (NSString *)modelPath {
    return [NSString stringWithCString:gpt_params.model.c_str() encoding:NSUTF8StringEncoding];
}

- (void)setModelPath:(NSString *)modelPath {
    gpt_params.model = [modelPath cStringUsingEncoding:NSUTF8StringEncoding];
}

- (NSString *)pathPromptCache {
    return [[NSString alloc] initWithCString:gpt_params.path_prompt_cache.c_str() encoding:NSUTF8StringEncoding];
}

- (void)setPathPromptCache:(NSString *)pathPromptCache {
    gpt_params.path_prompt_cache = [pathPromptCache cStringUsingEncoding:NSUTF8StringEncoding];
}

- (BOOL)enableChatTemplate {
    return gpt_params.enable_chat_template;
}

- (void)setEnableChatTemplate:(BOOL)enableChatTemplate {
    gpt_params.enable_chat_template = enableChatTemplate;
}

- (NSString *)chatTemplate {
    return [NSString stringWithCString:gpt_params.chat_template.c_str()
                              encoding:NSUTF8StringEncoding];
}

- (void)setChatTemplate:(NSString *)chatTemplate {
    gpt_params.chat_template = [chatTemplate cStringUsingEncoding:NSUTF8StringEncoding];
}

- (NSString *)inputPrefix {
    return [NSString stringWithCString:gpt_params.input_prefix.c_str()
                              encoding:NSUTF8StringEncoding];
}

- (void)setInputPrefix:(NSString *)inputPrefix {
    gpt_params.input_prefix = [inputPrefix cStringUsingEncoding:NSUTF8StringEncoding];
}

- (NSString *)inputSuffix {
    return [NSString stringWithCString:gpt_params.input_suffix.c_str()
                              encoding:NSUTF8StringEncoding];
}

- (void)setInputSuffix:(NSString *)inputSuffix {
    gpt_params.input_suffix = [inputSuffix cStringUsingEncoding:NSUTF8StringEncoding];
}

- (BOOL)interactive {
    return gpt_params.interactive;
}

- (void)setInteractive:(BOOL)interactive {
    gpt_params.interactive = interactive;
}

- (BOOL)interactiveFirst {
    return gpt_params.interactive_first;
}

- (void)setInteractiveFirst:(BOOL)interactiveFirst {
    gpt_params.interactive_first = interactiveFirst;
}
- (id)copyWithZone:(NSZone *)zone {
    GPTParams *copy = [[[self class] allocWithZone:zone] init];
    
    if (copy) {
        copy->gpt_params = gpt_params;
    }
    
    return copy;
}

@end
