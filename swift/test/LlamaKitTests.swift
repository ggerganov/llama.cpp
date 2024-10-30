import Foundation
import Testing
@testable import LlamaKit
import JSONSchema

// MARK: LlamaGrammarSession Suite
@Suite("LlamaGrammarSession Suite")
struct LlamaGrammarSessionSuite {
    @JSONSchema struct Trip {
        let location: String
        let startDate: TimeInterval
        let durationInDays: Int
    }

    func downloadFile() async throws -> String {
        let fm = FileManager.default
        let tmpDir = fm.temporaryDirectory
        let destinationURL = tmpDir.appending(path: "tinyllama.gguf")
        
        guard !fm.fileExists(atPath: destinationURL.path()) else {
            return destinationURL.path()
        }
        print("Downloading TinyLlama, this may take a while...")
        // Define the URL
        guard let url = URL(string: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf?download=true") else {
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
    
    @Test func llamaGrammarSession() async throws {
        let params = GPTParams()
        params.modelPath = try await downloadFile()
        params.nPredict = 256
        params.nCtx = 1024
        params.cpuParams.nThreads = 4
        params.cpuParamsBatch.nThreads = 4
        params.nBatch = 1024
        params.nGpuLayers = 128
        params.chatTemplate = """
        <|system|>
        {system_message}</s>
        <|user|>
        {prompt}</s>
        <|assistant|>
        """
        params.prompt = """
        You are a travel agent. The current date epoch \(Date.now.timeIntervalSince1970).
        Responses should have the following fields:
        
            location: the location of the trip
            startDate: the start of the trip as the unix epoch since 1970
            durationInDays: the duration of the trip in days
        
        """
        params.interactive = true
        let session = try await LlamaSession<Trip>(params: params)
        await #expect(throws: Never.self) {
            let trip = try await session.chat(message: "Please create a trip for me to New York City that starts two weeks from now. The duration of the trip MUST be 3 days long.")
            #expect(trip.location.contains("New York"))
            // TODO: Testing the other fields is difficult considering model size
            // TODO: so for now, we are just asserting the grammar works
        }
    }
}

import WeatherKit
import CoreLocation

@llamaActor actor MyLlama {
    struct CurrentWeather: Codable {
        let temperature: Double
        let condition: WeatherCondition
    }
    
    /// Get the current weather in a given location.
    /// - parameter location: The city and state, e.g. San Francisco, CA
    /// - parameter unit: The unit of temperature
    public static func getCurrentWeather(location: String, unit: String) async throws -> CurrentWeather {
        let weather = try await WeatherService().weather(for: CLGeocoder().geocodeAddressString(location)[0].location!)
        var temperature = weather.currentWeather.temperature
        temperature.convert(to: .fahrenheit)
        return CurrentWeather(temperature: temperature.value,
                              condition: weather.currentWeather.condition)
    }
}

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
    let currentWeather = try await MyLlama.getCurrentWeather(location: "San Francisco, CA", unit: "farenheit")
    let output = try await llama.chat("What's the weather (in farenheit) in San Francisco, CA?")
    #expect(output.contains(String(format: "%.2f", currentWeather.temperature)))
    // #expect(output.contains(currentWeather.condition.rawValue))
}
