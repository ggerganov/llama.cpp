#ifndef GPTParams_h
#define GPTParams_h

@class LlamaModelParams;
@class LlamaContextParams;
@class GGMLThreadpool;
@class CPUParams;

@interface GPTSamplerParams : NSObject

// Properties corresponding to C++ struct fields
@property (nonatomic, assign) uint32_t seed;
@property (nonatomic, assign) int32_t nPrev;
@property (nonatomic, assign) int32_t nProbs;
@property (nonatomic, assign) int32_t minKeep;
@property (nonatomic, assign) int32_t topK;
@property (nonatomic, assign) float topP;
@property (nonatomic, assign) float minP;
@property (nonatomic, assign) float tfsZ;
@property (nonatomic, assign) float typP;
@property (nonatomic, assign) float temp;
@property (nonatomic, assign) float dynatempRange;
@property (nonatomic, assign) float dynatempExponent;
@property (nonatomic, assign) int32_t penaltyLastN;
@property (nonatomic, assign) float penaltyRepeat;
@property (nonatomic, assign) float penaltyFreq;
@property (nonatomic, assign) float penaltyPresent;
@property (nonatomic, assign) int32_t mirostat;
@property (nonatomic, assign) float mirostatTau;
@property (nonatomic, assign) float mirostatEta;
@property (nonatomic, assign) BOOL penalizeNl;
@property (nonatomic, assign) BOOL ignoreEos;
@property (nonatomic, assign) BOOL noPerf;
@property (nonatomic, strong) NSArray<NSNumber *> *samplers;
@property (nonatomic, copy) NSString *grammar;
@property (nonatomic, strong) NSArray<NSNumber *> *logitBias;

- (NSString *)print;

@end

@interface GPTParams : NSObject

@property (nonatomic, assign) int32_t nPredict;
@property (nonatomic, assign) NSInteger nCtx;
@property (nonatomic, assign) int32_t nBatch;
@property (nonatomic, assign) int32_t nUBatch;
@property (nonatomic, assign) int32_t nKeep;
@property (nonatomic, assign) int32_t nDraft;
@property (nonatomic, assign) int32_t nChunks;
@property (nonatomic, assign) int32_t nParallel;
@property (nonatomic, assign) int32_t nSequences;
@property (nonatomic, assign) float pSplit;
@property (nonatomic, assign) int32_t nGpuLayers;
@property (nonatomic, assign) int32_t nGpuLayersDraft;
@property (nonatomic, assign) int32_t mainGpu;
@property (nonatomic, strong) NSArray<NSNumber *> *tensorSplit;
@property (nonatomic, assign) int32_t grpAttnN;
@property (nonatomic, assign) int32_t grpAttnW;
@property (nonatomic, assign) int32_t nPrint;
@property (nonatomic, assign) float ropeFreqBase;
@property (nonatomic, assign) float ropeFreqScale;
@property (nonatomic, assign) float yarnExtFactor;
@property (nonatomic, assign) float yarnAttnFactor;
@property (nonatomic, assign) float yarnBetaFast;
@property (nonatomic, assign) float yarnBetaSlow;
@property (nonatomic, assign) int32_t yarnOrigCtx;
@property (nonatomic, assign) float defragThold;

@property (nonatomic, strong) CPUParams *cpuParams;
@property (nonatomic, strong) CPUParams *cpuParamsBatch;
@property (nonatomic, strong) CPUParams *draftCpuParams;
@property (nonatomic, strong) CPUParams *draftCpuParamsBatch;

@property (nonatomic, copy) void (^cbEval)(void *);
@property (nonatomic, assign) void *cbEvalUserData;

@property (nonatomic, assign) NSInteger numaStrategy; // Enumerations

@property (nonatomic, assign) NSInteger splitMode;
@property (nonatomic, assign) NSInteger ropeScalingType;
@property (nonatomic, assign) NSInteger poolingType;
@property (nonatomic, assign) NSInteger attentionType;

// Sampler parameters would also be converted to an Objective-C object
@property (nonatomic, strong) GPTSamplerParams *samplerParams;

@property (nonatomic, copy) NSString *modelPath;
@property (nonatomic, copy) NSString *modelDraft;
@property (nonatomic, copy) NSString *modelAlias;
@property (nonatomic, copy) NSString *modelURL;
@property (nonatomic, copy) NSString *hfToken;
@property (nonatomic, copy) NSString *hfRepo;
@property (nonatomic, copy) NSString *hfFile;
@property (nonatomic, copy) NSString *prompt;
@property (nonatomic, copy) NSString *promptFile;
@property (nonatomic, copy) NSString *pathPromptCache;
@property (nonatomic, copy) NSString *inputPrefix;
@property (nonatomic, copy) NSString *inputSuffix;
@property (nonatomic, copy) NSString *logdir;
@property (nonatomic, copy) NSString *lookupCacheStatic;
@property (nonatomic, copy) NSString *lookupCacheDynamic;
@property (nonatomic, copy) NSString *logitsFile;
@property (nonatomic, copy) NSString *rpcServers;

@property (nonatomic, strong) NSArray<NSString *> *inputFiles;
@property (nonatomic, strong) NSArray<NSString *> *antiPrompts;
@property (nonatomic, strong) NSArray *kvOverrides;

@property (nonatomic, assign) BOOL loraInitWithoutApply;
@property (nonatomic, strong) NSArray *loraAdapters;
@property (nonatomic, strong) NSArray *controlVectors;

// Control params
@property (nonatomic, assign) int32_t verbosity;
@property (nonatomic, assign) int32_t controlVectorLayerStart;
@property (nonatomic, assign) int32_t controlVectorLayerEnd;

// Performance and configuration params
@property (nonatomic, assign) int32_t pplStride;
@property (nonatomic, assign) int32_t pplOutputType;

@property (nonatomic, assign) BOOL hellaswag;
@property (nonatomic, assign) size_t hellaswagTasks;
@property (nonatomic, assign) BOOL winogrande;
@property (nonatomic, assign) size_t winograndeTasks;
@property (nonatomic, assign) BOOL multipleChoice;
@property (nonatomic, assign) size_t multipleChoiceTasks;
@property (nonatomic, assign) BOOL klDivergence;

@property (nonatomic, assign) BOOL usage;
@property (nonatomic, assign) BOOL useColor;
@property (nonatomic, assign) BOOL special;
@property (nonatomic, assign) BOOL interactive;
@property (nonatomic, assign) BOOL interactiveFirst;
@property (nonatomic, assign) BOOL conversation;
@property (nonatomic, assign) BOOL promptCacheAll;
@property (nonatomic, assign) BOOL promptCacheRO;

@property (nonatomic, assign) BOOL escapeSequences;
@property (nonatomic, assign) BOOL multilineInput;
@property (nonatomic, assign) BOOL simpleIO;
@property (nonatomic, assign) BOOL continuousBatching;
@property (nonatomic, assign) BOOL flashAttention;
@property (nonatomic, assign) BOOL noPerformanceMetrics;
@property (nonatomic, assign) BOOL contextShift;

// Server and I/O settings
@property (nonatomic, assign) int32_t port;
@property (nonatomic, assign) int32_t timeoutRead;
@property (nonatomic, assign) int32_t timeoutWrite;
@property (nonatomic, assign) int32_t httpThreads;

@property (nonatomic, copy) NSString *hostname;
@property (nonatomic, copy) NSString *publicPath;
@property (nonatomic, copy) NSString *chatTemplate;
@property (nonatomic, copy) NSString *systemPrompt;
@property (nonatomic, assign) BOOL enableChatTemplate;

@property (nonatomic, strong) NSArray<NSString *> *apiKeys;

@property (nonatomic, copy) NSString *sslFileKey;
@property (nonatomic, copy) NSString *sslFileCert;

@property (nonatomic, assign) BOOL endpointSlots;
@property (nonatomic, assign) BOOL endpointMetrics;
@property (nonatomic, assign) BOOL logJSON;

@property (nonatomic, copy) NSString *slotSavePath;
@property (nonatomic, assign) float slotPromptSimilarity;

// batched-bench params
@property (nonatomic, assign) BOOL isPPShared;
@property (nonatomic, strong) NSArray<NSNumber *> *nPP;
@property (nonatomic, strong) NSArray<NSNumber *> *nTG;
@property (nonatomic, strong) NSArray<NSNumber *> *nPL;

// retrieval params
@property (nonatomic, strong) NSArray<NSString *> *contextFiles;
@property (nonatomic, assign) int32_t chunkSize;
@property (nonatomic, copy) NSString *chunkSeparator;

// passkey params
@property (nonatomic, assign) int32_t nJunk;
@property (nonatomic, assign) int32_t iPos;

// imatrix params
@property (nonatomic, copy) NSString *outFile;
@property (nonatomic, assign) int32_t nOutFreq;
@property (nonatomic, assign) int32_t nSaveFreq;
@property (nonatomic, assign) int32_t iChunk;
@property (nonatomic, assign) BOOL processOutput;
@property (nonatomic, assign) BOOL computePPL;

// cvector-generator params
@property (nonatomic, assign) int nPCABatch;
@property (nonatomic, assign) int nPCAIterations;
@property (nonatomic, assign) int cvectorDimreMethod;
@property (nonatomic, copy) NSString *cvectorOutfile;
@property (nonatomic, copy) NSString *cvectorPositiveFile;
@property (nonatomic, copy) NSString *cvectorNegativeFile;

@property (nonatomic, assign) BOOL spmInfill;
@property (nonatomic, copy) NSString *loraOutfile;
@property (nonatomic, assign) BOOL embedding;
@property (nonatomic, assign) BOOL verbosePrompt; // print prompt tokens before generation
@property (nonatomic, assign) BOOL batchedBenchOutputJSONL;
@property (nonatomic, assign) BOOL inputPrefixBOS; // prefix BOS to user inputs, preceding input_prefix
@property (nonatomic, assign) BOOL ctxShift; // context shift on inifinite text generation
@property (nonatomic, assign) BOOL displayPrompt; // print prompt before generation

@property (nonatomic, assign) BOOL logging; // print logging

@end

#endif /* GPTParams_h */
