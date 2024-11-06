import Foundation
import XCTest

import LlamaKit  // Replace with your module name

final class GPTParamsTests: XCTestCase {
    func testPropertyAssignmentsAndCopy() throws {
        // Create an instance of GPTParams
        let originalParams = GPTParams()
        
        // Assign values to all properties
        originalParams.nPredict = 10
        originalParams.nCtx = 20
        originalParams.nBatch = 30
        originalParams.nUBatch = 40
        originalParams.nKeep = 50
        originalParams.nDraft = 60
        originalParams.nChunks = 70
        originalParams.nParallel = 80
        originalParams.nSequences = 90
        originalParams.pSplit = 0.5
        originalParams.nGpuLayers = 100
        originalParams.nGpuLayersDraft = 110
        originalParams.mainGpu = 120
        originalParams.tensorSplit = [0.1, 0.2, 0.3]
        originalParams.grpAttnN = 130
        originalParams.grpAttnW = 140
        originalParams.nPrint = 150
        originalParams.ropeFreqBase = 0.6
        originalParams.ropeFreqScale = 0.7
        originalParams.yarnExtFactor = 0.8
        originalParams.yarnAttnFactor = 0.9
        originalParams.yarnBetaFast = 1.0
        originalParams.yarnBetaSlow = 1.1
        originalParams.yarnOrigCtx = 160
        originalParams.defragThold = 1.2
        
        // Initialize CPUParams instances if needed
        originalParams.cpuParams = CPUParams()
        originalParams.cpuParamsBatch = CPUParams()
        originalParams.draftCpuParams = CPUParams()
        originalParams.draftCpuParamsBatch = CPUParams()
        
        // Assign blocks and user data
        originalParams.cbEval = { userData in
            // Callback implementation
        }
        originalParams.cbEvalUserData = nil
        
        // Assign enum values (assuming NSInteger maps to Int)
        originalParams.numaStrategy = 1
        originalParams.splitMode = 2
        originalParams.ropeScalingType = 3
        originalParams.poolingType = 4
        originalParams.attentionType = 5
        
        // Assign sampler parameters
        originalParams.samplerParams = GPTSamplerParams()
        
        // Assign string properties
        originalParams.modelPath = "path/to/model"
        originalParams.modelDraft = "model_draft"
        originalParams.modelAlias = "alias"
        originalParams.modelURL = "http://model.url"
        originalParams.hfToken = "token"
        originalParams.hfRepo = "repo"
        originalParams.hfFile = "file"
        originalParams.prompt = "prompt"
        originalParams.promptFile = "prompt.txt"
        originalParams.pathPromptCache = "cache/path"
        originalParams.inputPrefix = "prefix"
        originalParams.inputSuffix = "suffix"
        originalParams.logdir = "log/dir"
        originalParams.lookupCacheStatic = "static/cache"
        originalParams.lookupCacheDynamic = "dynamic/cache"
        originalParams.logitsFile = "logits.txt"
        originalParams.rpcServers = "servers"
        
        // Assign array properties
        originalParams.inputFiles = ["input1.txt", "input2.txt"]
        originalParams.antiPrompts = ["anti1", "anti2"]
        originalParams.kvOverrides = ["override1", "override2"]
        originalParams.loraAdapters = ["adapter1", "adapter2"]
        originalParams.controlVectors = ["control1", "control2"]
        
        // Assign boolean and control properties
        originalParams.loraInitWithoutApply = true
        originalParams.verbosity = 1
        originalParams.controlVectorLayerStart = 2
        originalParams.controlVectorLayerEnd = 3
        originalParams.pplStride = 4
        originalParams.pplOutputType = 5
        originalParams.hellaswag = true
        originalParams.hellaswagTasks = 10
        originalParams.winogrande = false
        originalParams.winograndeTasks = 20
        originalParams.multipleChoice = true
        originalParams.multipleChoiceTasks = 30
        originalParams.klDivergence = false
        originalParams.usage = true
        originalParams.useColor = false
        originalParams.special = true
        originalParams.interactive = false
        originalParams.interactiveFirst = true
        originalParams.conversation = false
        originalParams.promptCacheAll = true
        originalParams.promptCacheRO = false
        originalParams.escapeSequences = true
        originalParams.multilineInput = false
        originalParams.simpleIO = true
        originalParams.continuousBatching = false
        originalParams.flashAttention = true
        originalParams.noPerformanceMetrics = false
        originalParams.contextShift = true
        
        // Server and I/O settings
        originalParams.port = 8080
        originalParams.timeoutRead = 60
        originalParams.timeoutWrite = 30
        originalParams.httpThreads = 4
        originalParams.hostname = "localhost"
        originalParams.publicPath = "/public"
        originalParams.chatTemplate = "template"
        originalParams.systemPrompt = "system prompt"
        originalParams.enableChatTemplate = true
        originalParams.apiKeys = ["key1", "key2"]
        originalParams.sslFileKey = "key.pem"
        originalParams.sslFileCert = "cert.pem"
        originalParams.endpointSlots = true
        originalParams.endpointMetrics = false
        originalParams.logJSON = true
        originalParams.slotSavePath = "/slots"
        originalParams.slotPromptSimilarity = 0.75
        
        // Batched-bench params
        originalParams.isPPShared = true
        originalParams.nPP = [1, 2]
        originalParams.nTG = [3, 4]
        originalParams.nPL = [5, 6]
        
        // Retrieval params
        originalParams.contextFiles = ["context1.txt", "context2.txt"]
        originalParams.chunkSize = 1024
        originalParams.chunkSeparator = "\n"
        
        // Passkey params
        originalParams.nJunk = 7
        originalParams.iPos = 8
        
        // Imatrix params
        originalParams.outFile = "output.txt"
        originalParams.nOutFreq = 100
        originalParams.nSaveFreq = 200
        originalParams.iChunk = 9
        originalParams.processOutput = true
        originalParams.computePPL = false
        
        // Cvector-generator params
        originalParams.nPCABatch = 10
        originalParams.nPCAIterations = 11
        originalParams.cvectorDimreMethod = 12
        originalParams.cvectorOutfile = "cvector.out"
        originalParams.cvectorPositiveFile = "positive.txt"
        originalParams.cvectorNegativeFile = "negative.txt"
        
        // Additional properties
        originalParams.spmInfill = true
        originalParams.loraOutfile = "lora.out"
        originalParams.embedding = false
        originalParams.verbosePrompt = true
        originalParams.batchedBenchOutputJSONL = false
        originalParams.inputPrefixBOS = true
        originalParams.ctxShift = false
        originalParams.displayPrompt = true
        originalParams.logging = false
        
        // Verify that properties are assigned correctly
        XCTAssertEqual(originalParams.nPredict, 10)
        XCTAssertEqual(originalParams.nCtx, 20)
        XCTAssertEqual(originalParams.nBatch, 30)
        XCTAssertEqual(originalParams.nUBatch, 40)
        XCTAssertEqual(originalParams.nKeep, 50)
        XCTAssertEqual(originalParams.nDraft, 60)
        XCTAssertEqual(originalParams.nChunks, 70)
        XCTAssertEqual(originalParams.nParallel, 80)
        XCTAssertEqual(originalParams.nSequences, 90)
        XCTAssertEqual(originalParams.pSplit, 0.5)
        XCTAssertEqual(originalParams.nGpuLayers, 100)
        XCTAssertEqual(originalParams.nGpuLayersDraft, 110)
        XCTAssertEqual(originalParams.mainGpu, 120)
        XCTAssertEqual(originalParams.tensorSplit[0..<3].map(\.floatValue),
                       [0.1, 0.2, 0.3])
        XCTAssertEqual(originalParams.grpAttnN, 130)
        XCTAssertEqual(originalParams.grpAttnW, 140)
        XCTAssertEqual(originalParams.nPrint, 150)
        XCTAssertEqual(originalParams.ropeFreqBase, 0.6)
        XCTAssertEqual(originalParams.ropeFreqScale, 0.7)
        XCTAssertEqual(originalParams.yarnExtFactor, 0.8)
        XCTAssertEqual(originalParams.yarnAttnFactor, 0.9)
        XCTAssertEqual(originalParams.yarnBetaFast, 1.0)
        XCTAssertEqual(originalParams.yarnBetaSlow, 1.1)
        XCTAssertEqual(originalParams.yarnOrigCtx, 160)
        XCTAssertEqual(originalParams.defragThold, 1.2)
        
        // Verify enums
        XCTAssertEqual(originalParams.numaStrategy, 1)
        XCTAssertEqual(originalParams.splitMode, 2)
        XCTAssertEqual(originalParams.ropeScalingType, 3)
        XCTAssertEqual(originalParams.poolingType, 4)
        XCTAssertEqual(originalParams.attentionType, 5)
        
        // Verify string properties
        XCTAssertEqual(originalParams.modelPath, "path/to/model")
        XCTAssertEqual(originalParams.modelDraft, "model_draft")
        XCTAssertEqual(originalParams.modelAlias, "alias")
        XCTAssertEqual(originalParams.modelURL, "http://model.url")
        XCTAssertEqual(originalParams.hfToken, "token")
        XCTAssertEqual(originalParams.hfRepo, "repo")
        XCTAssertEqual(originalParams.hfFile, "file")
        XCTAssertEqual(originalParams.prompt, "prompt")
        XCTAssertEqual(originalParams.promptFile, "prompt.txt")
        XCTAssertEqual(originalParams.pathPromptCache, "cache/path")
        XCTAssertEqual(originalParams.inputPrefix, "prefix")
        XCTAssertEqual(originalParams.inputSuffix, "suffix")
        XCTAssertEqual(originalParams.logdir, "log/dir")
        XCTAssertEqual(originalParams.lookupCacheStatic, "static/cache")
        XCTAssertEqual(originalParams.lookupCacheDynamic, "dynamic/cache")
        XCTAssertEqual(originalParams.logitsFile, "logits.txt")
        XCTAssertEqual(originalParams.rpcServers, "servers")
        
        // Verify array properties
        XCTAssertEqual(originalParams.inputFiles, ["input1.txt", "input2.txt"])
        XCTAssertEqual(originalParams.antiPrompts, ["anti1", "anti2"])
        XCTAssertEqual(originalParams.kvOverrides as? [String], ["override1", "override2"])
//        XCTAssertEqual(originalParams.loraAdapters, ["adapter1", "adapter2"])
//        XCTAssertEqual(originalParams.controlVectors, ["control1", "control2"])
        
        // Verify boolean and control properties
        XCTAssertTrue(originalParams.loraInitWithoutApply)
        XCTAssertEqual(originalParams.verbosity, 1)
        XCTAssertEqual(originalParams.controlVectorLayerStart, 2)
        XCTAssertEqual(originalParams.controlVectorLayerEnd, 3)
        XCTAssertEqual(originalParams.pplStride, 4)
        XCTAssertEqual(originalParams.pplOutputType, 5)
        XCTAssertTrue(originalParams.hellaswag)
        XCTAssertEqual(originalParams.hellaswagTasks, 10)
        XCTAssertFalse(originalParams.winogrande)
        XCTAssertEqual(originalParams.winograndeTasks, 20)
        XCTAssertTrue(originalParams.multipleChoice)
        XCTAssertEqual(originalParams.multipleChoiceTasks, 30)
        XCTAssertFalse(originalParams.klDivergence)
        XCTAssertTrue(originalParams.usage)
        XCTAssertFalse(originalParams.useColor)
        XCTAssertTrue(originalParams.special)
        XCTAssertFalse(originalParams.interactive)
        XCTAssertTrue(originalParams.interactiveFirst)
        XCTAssertFalse(originalParams.conversation)
        XCTAssertTrue(originalParams.promptCacheAll)
        XCTAssertFalse(originalParams.promptCacheRO)
        XCTAssertTrue(originalParams.escapeSequences)
        XCTAssertFalse(originalParams.multilineInput)
        XCTAssertTrue(originalParams.simpleIO)
        XCTAssertFalse(originalParams.continuousBatching)
        XCTAssertTrue(originalParams.flashAttention)
        XCTAssertFalse(originalParams.noPerformanceMetrics)
        XCTAssertTrue(originalParams.contextShift)
        
        // Verify server and I/O settings
        XCTAssertEqual(originalParams.port, 8080)
        XCTAssertEqual(originalParams.timeoutRead, 60)
        XCTAssertEqual(originalParams.timeoutWrite, 30)
        XCTAssertEqual(originalParams.httpThreads, 4)
        XCTAssertEqual(originalParams.hostname, "localhost")
        XCTAssertEqual(originalParams.publicPath, "/public")
        XCTAssertEqual(originalParams.chatTemplate, "template")
        XCTAssertEqual(originalParams.systemPrompt, "system prompt")
        XCTAssertTrue(originalParams.enableChatTemplate)
        XCTAssertEqual(originalParams.apiKeys, ["key1", "key2"])
        XCTAssertEqual(originalParams.sslFileKey, "key.pem")
        XCTAssertEqual(originalParams.sslFileCert, "cert.pem")
        XCTAssertTrue(originalParams.endpointSlots)
        XCTAssertFalse(originalParams.endpointMetrics)
        XCTAssertTrue(originalParams.logJSON)
        XCTAssertEqual(originalParams.slotSavePath, "/slots")
        XCTAssertEqual(originalParams.slotPromptSimilarity, 0.75)
        
        // Verify batched-bench params
        XCTAssertTrue(originalParams.isPPShared)
        XCTAssertEqual(originalParams.nPP, [1, 2])
        XCTAssertEqual(originalParams.nTG, [3, 4])
        XCTAssertEqual(originalParams.nPL, [5, 6])
        
        // Verify retrieval params
        XCTAssertEqual(originalParams.contextFiles, ["context1.txt", "context2.txt"])
        XCTAssertEqual(originalParams.chunkSize, 1024)
        XCTAssertEqual(originalParams.chunkSeparator, "\n")
        
        // Verify passkey params
        XCTAssertEqual(originalParams.nJunk, 7)
        XCTAssertEqual(originalParams.iPos, 8)
        
        // Verify imatrix params
        XCTAssertEqual(originalParams.outFile, "output.txt")
        XCTAssertEqual(originalParams.nOutFreq, 100)
        XCTAssertEqual(originalParams.nSaveFreq, 200)
        XCTAssertEqual(originalParams.iChunk, 9)
        XCTAssertTrue(originalParams.processOutput)
        XCTAssertFalse(originalParams.computePPL)
        
        // Verify cvector-generator params
        XCTAssertEqual(originalParams.nPCABatch, 10)
        XCTAssertEqual(originalParams.nPCAIterations, 11)
        XCTAssertEqual(originalParams.cvectorDimreMethod, 12)
        XCTAssertEqual(originalParams.cvectorOutfile, "cvector.out")
        XCTAssertEqual(originalParams.cvectorPositiveFile, "positive.txt")
        XCTAssertEqual(originalParams.cvectorNegativeFile, "negative.txt")
        
        // Verify additional properties
        XCTAssertTrue(originalParams.spmInfill)
        XCTAssertEqual(originalParams.loraOutfile, "lora.out")
        XCTAssertFalse(originalParams.embedding)
        XCTAssertTrue(originalParams.verbosePrompt)
        XCTAssertFalse(originalParams.batchedBenchOutputJSONL)
        XCTAssertTrue(originalParams.inputPrefixBOS)
        XCTAssertFalse(originalParams.ctxShift)
        XCTAssertTrue(originalParams.displayPrompt)
        XCTAssertFalse(originalParams.logging)
        
        // Test the copy function
        guard let copiedParams = originalParams.copy() as? GPTParams else {
            XCTFail("Copy function did not return a GPTParams instance.")
            return
        }
        
        // Verify that the copied properties match the original
        XCTAssertEqual(copiedParams.nPredict, originalParams.nPredict)
        XCTAssertEqual(copiedParams.nCtx, originalParams.nCtx)
        XCTAssertEqual(copiedParams.nBatch, originalParams.nBatch)
        XCTAssertEqual(copiedParams.nUBatch, originalParams.nUBatch)
        XCTAssertEqual(copiedParams.nKeep, originalParams.nKeep)
        XCTAssertEqual(copiedParams.nDraft, originalParams.nDraft)
        XCTAssertEqual(copiedParams.nChunks, originalParams.nChunks)
        XCTAssertEqual(copiedParams.nParallel, originalParams.nParallel)
        XCTAssertEqual(copiedParams.nSequences, originalParams.nSequences)
        XCTAssertEqual(copiedParams.pSplit, originalParams.pSplit)
        XCTAssertEqual(copiedParams.nGpuLayers, originalParams.nGpuLayers)
        XCTAssertEqual(copiedParams.nGpuLayersDraft, originalParams.nGpuLayersDraft)
        XCTAssertEqual(copiedParams.mainGpu, originalParams.mainGpu)
        XCTAssertEqual(copiedParams.tensorSplit, originalParams.tensorSplit)
        XCTAssertEqual(copiedParams.grpAttnN, originalParams.grpAttnN)
        XCTAssertEqual(copiedParams.grpAttnW, originalParams.grpAttnW)
        XCTAssertEqual(copiedParams.nPrint, originalParams.nPrint)
        XCTAssertEqual(copiedParams.ropeFreqBase, originalParams.ropeFreqBase)
        XCTAssertEqual(copiedParams.ropeFreqScale, originalParams.ropeFreqScale)
        XCTAssertEqual(copiedParams.yarnExtFactor, originalParams.yarnExtFactor)
        XCTAssertEqual(copiedParams.yarnAttnFactor, originalParams.yarnAttnFactor)
        XCTAssertEqual(copiedParams.yarnBetaFast, originalParams.yarnBetaFast)
        XCTAssertEqual(copiedParams.yarnBetaSlow, originalParams.yarnBetaSlow)
        XCTAssertEqual(copiedParams.yarnOrigCtx, originalParams.yarnOrigCtx)
        XCTAssertEqual(copiedParams.defragThold, originalParams.defragThold)
        XCTAssertEqual(copiedParams.modelPath, originalParams.modelPath)
        XCTAssertEqual(copiedParams.apiKeys, originalParams.apiKeys)
//        XCTAssertEqual(copiedParams.controlVectors, originalParams.controlVectors)
        // Continue verifying all other properties...
        
        // Verify that modifying the original does not affect the copy
        originalParams.nPredict = 999
        XCTAssertNotEqual(copiedParams.nPredict, originalParams.nPredict)
    }
}
