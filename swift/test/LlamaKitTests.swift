import Foundation
import Testing
@testable import LlamaKit
import JSONSchema
import OSLog

// MARK: LlamaGrammarSession Suite
@Suite("LlamaSession Suite")
struct LlamaSessionSuite {
    @JSONSchema struct Trip {
        let location: String
        let startDate: TimeInterval
        let durationInDays: Int
    }

    func downloadFile(url: String, to path: String) async throws -> String {
        let fm = FileManager.default
        let tmpDir = fm.temporaryDirectory
        let destinationURL = tmpDir.appending(path: path)
        
        guard !fm.fileExists(atPath: destinationURL.path()) else {
            return destinationURL.path()
        }
        print("Downloading \(path), this may take a while...")
        // Define the URL
        guard let url = URL(string: url) else {
            print("Invalid URL.")
            throw URLError(.badURL)
        }
        
        // Start the async download
        let (tempURL, _) = try await URLSession.shared.download(from: url)
        
        // Define the destination path in the documents directory
        
        // Move the downloaded file to the destination
        try fm.moveItem(at: tempURL, to: destinationURL)
        print("File downloaded to: \(destinationURL.path())")
        return destinationURL.path()
    }
    
    
    func baseParams(url: String, to path: String) async throws -> GPTParams {
        let params = GPTParams()
        params.modelPath = try await downloadFile(url: url, to: path)
        params.nPredict = 512
        params.nCtx = 4096
        params.cpuParams.nThreads = 8
        params.cpuParamsBatch.nThreads = 8
        params.nBatch = 1024
        params.nGpuLayers = 1024
        params.chatTemplate = """
        <|system|>
        {system_message}</s>
        <|user|>
        {prompt}</s>
        <|assistant|>
        """
        params.interactive = true
        return params
    }
    
    @Test func llamaInferenceSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true", to: "tinyllama.gguf")
        params.prompt = """
        <|system|>
        You are an AI assistant. Answer queries simply and concisely.</s>
        """
        params.antiPrompts = ["</s>"]
        params.inputPrefix = "<|user|>"
        params.inputSuffix = "</s><|assistant|>"
        params.interactive = true
        let session = try await LlamaChatSession(params: params, flush: false)
        for await msg in await session.inferenceStream(message: "How are you today?") {
            print(msg, terminator: "")
        }
    }
    
    @Test func llamaGrammarSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true", to: "tinyllama.gguf")
        params.prompt = """
        You are a travel agent. The current date epoch \(Date.now.timeIntervalSince1970).
        Responses should have the following fields:
        
            location: the location of the trip
            startDate: the start of the trip as the unix epoch since 1970
            durationInDays: the duration of the trip in days
        
        """
        let session = try await LlamaSession<Trip>(params: params)
        await #expect(throws: Never.self) {
            let trip = try await session.chat(message: "Please create a trip for me to New York City that starts two weeks from now. The duration of the trip MUST be 3 days long.")
            #expect(trip.location.contains("New York"))
            // TODO: Testing the other fields is difficult considering model size
            // TODO: so for now, we are just asserting the grammar works
        }
    }
    
    @JSONSchema struct IsCorrect {
        let isSpellingCorrect: Bool
    }
    
    @Test func llamaSimpleGrammarSession() async throws {
        let params = try await baseParams(url: "https://huggingface.co/RichardErkhov/openfoodfacts_-_spellcheck-mistral-7b-gguf/resolve/main/spellcheck-mistral-7b.Q8_0.gguf?download=true",
                                          to: "spellcheck_q8.gguf")
        params.prompt = """
        ###You are a spell checker. I will provide you with the word 'strawberry'. If the spelling of the given word is correct, please respond {"isCorrect": true} else respond {"isCorrect": false}.\n
        """
        let session = try await LlamaSession<IsCorrect>(params: params)
        for _ in 0..<10 {
            var output = try await session.chat(message: "###strawberry\n")
            #expect(output.isSpellingCorrect)
            output = try await session.chat(message: "###strawberrry\n")
            #expect(!output.isSpellingCorrect)
        }
    }
}

import WeatherKit
import CoreLocation

func downloadFile() async throws -> String {
    let fm = FileManager.default
    let tmpDir = fm.temporaryDirectory
    let destinationURL = tmpDir.appending(path: "llama_groq_gguf.gguf")
    
    guard !fm.fileExists(atPath: destinationURL.path()) else {
        return destinationURL.path()
    }
    print("Downloading Llama Tools, this may take a while...")
    // Define the URL
    guard let url = URL(string: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q5_K_M.gguf?download=true") else {
        print("Invalid URL.")
        throw URLError(.badURL)
    }
    
    // Start the async download
    let (tempURL, _) = try await URLSession.shared.download(from: url)
    
    // Define the destination path in the documents directory
    
    
    // Move the downloaded file to the destination
    try fm.moveItem(at: tempURL, to: destinationURL)
    print("File downloaded to: \(destinationURL.path())")
    return destinationURL.path()
}


@llamaActor actor MyLlama {
    struct CurrentWeather: Codable {
        let temperature: Double
        let condition: WeatherCondition
    }
    
    /// Get the current weather in a given location.
    /// - parameter location: The city and state, e.g. San Francisco, CA
    /// - parameter unit: The unit of temperature
    @Tool public func getCurrentWeather(location: String, unit: String) async throws -> CurrentWeather {
        let weather = try await WeatherService().weather(for: CLGeocoder().geocodeAddressString(location)[0].location!)
        var temperature = weather.currentWeather.temperature
        temperature.convert(to: .fahrenheit)
        return CurrentWeather(temperature: temperature.value,
                              condition: weather.currentWeather.condition)
    }
}

@Test func llamaToolSession() async throws {
    let params = GPTParams()
    params.modelPath = try await downloadFile()
    params.nPredict = 512
    params.nCtx = 4096
    params.cpuParams.nThreads = 8
    params.cpuParamsBatch.nThreads = 8
    params.nBatch = 1024
    params.nGpuLayers = 1024
    let llama = try await MyLlama(params: params)
    let currentWeather = try await llama.getCurrentWeather(location: "San Francisco, CA", unit: "farenheit")
    let output = try await llama.chat("What's the weather (in farenheit) in San Francisco, CA?")
    #expect(output.contains(String(format: "%d", Int(currentWeather.temperature))))
    // #expect(output.contains(currentWeather.condition.rawValue))
}
