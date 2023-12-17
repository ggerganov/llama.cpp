import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()

    @State private var multiLineText = ""

    private static func cleanupModelCaches() {
        // Delete all models (*.gguf)
        let fileManager = FileManager.default
        let documentsUrl =  FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let fileURLs = try fileManager.contentsOfDirectory(at: documentsUrl, includingPropertiesForKeys: nil)
            for fileURL in fileURLs {
                if fileURL.pathExtension == "gguf" {
                    try fileManager.removeItem(at: fileURL)
                }
            }
        } catch {
            print("Error while enumerating files \(documentsUrl.path): \(error.localizedDescription)")
        }
    }

    var body: some View {
        VStack {
            // automatically scroll to bottom of text view
            ScrollView(.vertical, showsIndicators: true) {
                Text(llamaState.messageLog)
            }

            TextEditor(text: $multiLineText)
                .frame(height: 200)
                .padding()
                .border(Color.gray, width: 0.5)

            // add two buttons "Send" and "Bench" next to each other
            HStack {
                Button("Send") {
                    sendText()
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)

                Button("Bench") {
                    bench()
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }

            VStack {
                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TheBloke / TinyLlama-1.1B-1T-OpenOrca-GGUF (Q4_0)",
                    modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",
                    filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf"
                )
                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TheBloke / TinyLlama-1.1B-1T-OpenOrca-GGUF (Q8_0)",
                    modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q8_0.gguf?download=true",
                    filename: "tinyllama-1.1b-1t-openorca.Q8_0.gguf"
                )
                Button("Clear downloaded models") {
                    ContentView.cleanupModelCaches()
                    llamaState.cacheCleared = true
                }
            }
        }
        .padding()
    }

    func sendText() {
        Task {
            await llamaState.complete(text: multiLineText)
            multiLineText = ""
        }
    }

    func bench() {
        Task {
            await llamaState.bench()
        }
    }
}

//#Preview {
//    ContentView()
//}
