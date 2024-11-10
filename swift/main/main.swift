import LlamaKit
import WeatherKit
import CoreLocation

func downloadFile() async throws -> String {
    let fm = FileManager.default
    let tmpDir = fm.temporaryDirectory
    let destinationURL = tmpDir.appending(path: "llama_tools.gguf")
    
    guard !fm.fileExists(atPath: destinationURL.path()) else {
        return destinationURL.path()
    }
    print("Downloading Llama Tools, this may take a while...")
    // Define the URL
    guard let url = URL(string: "https://huggingface.co/bartowski/Llama-3-Groq-8B-Tool-Use-GGUF/resolve/main/Llama-3-Groq-8B-Tool-Use-Q8_0.gguf?download=true") else {
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
    /// Get the current date.
    @Tool public func getCurrentDate() -> String {
        Date.now.formatted(date: .long, time: .complete)
    }
}

let params = GPTParams()
params.modelPath = try await downloadFile()
params.nPredict = 512
params.nCtx = 4096
params.cpuParams.nThreads = 8
params.cpuParamsBatch.nThreads = 8
params.nBatch = 1024
params.nGpuLayers = 1024
let llama = try await MyLlama(params: params)

while true {
    print("Enter input: ", terminator: "")

    // Read user input
    if let userInput = readLine() {
        if userInput.lowercased() == "exit" {
            print("Exiting the loop.")
            break
        } else {
            print("🧔🏽‍♂️: \(userInput)")
            let response = try await llama.infer(userInput)
            print("🤖: \(response)")
        }
    } else {
        print("Failed to read input.")
    }
}
