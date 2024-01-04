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
            ScrollView(.vertical, showsIndicators: true) {
                Text(llamaState.messageLog)
                .font(.system(size: 12))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .onTapGesture {
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
            }

            TextEditor(text: $multiLineText)
                .frame(height: 80)
                .padding()
                .border(Color.gray, width: 0.5)

            HStack {
                Button("Send") {
                    sendText()
                }

                Button("Bench") {
                    bench()
                }

                Button("Clear") {
                    clear()
                }

                Button("Copy") {
                    UIPasteboard.general.string = llamaState.messageLog
                }
            }.buttonStyle(.bordered)

            VStack(alignment: .leading) {
                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",
                    modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",
                    filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf"
                )

                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TinyLlama-1.1B (Q8_0, 1.1 GiB)",
                    modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q8_0.gguf?download=true",
                    filename: "tinyllama-1.1b-1t-openorca.Q8_0.gguf"
                )

                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TinyLlama-1.1B (F16, 2.2 GiB)",
                    modelUrl: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
                    filename: "tinyllama-1.1b-f16.gguf"
                )

                DownloadButton(
                    llamaState: llamaState,
                    modelName: "Phi-2.7B (Q4_0, 1.6 GiB)",
                    modelUrl: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
                    filename: "phi-2-q4_0.gguf"
                )

                DownloadButton(
                    llamaState: llamaState,
                    modelName: "Phi-2.7B (Q8_0, 2.8 GiB)",
                    modelUrl: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
                    filename: "phi-2-q8_0.gguf"
                )

                DownloadButton(
                    llamaState: llamaState,
                    modelName: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
                    modelUrl: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
                    filename: "mistral-7b-v0.1.Q4_0.gguf"
                )

                Button("Clear downloaded models") {
                    ContentView.cleanupModelCaches()
                    llamaState.cacheCleared = true
                }

                LoadCustomButton(llamaState: llamaState)
            }
            .padding(.top, 4)
            .font(.system(size: 12))
            .frame(maxWidth: .infinity, alignment: .leading)
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

    func clear() {
        Task {
            await llamaState.clear()
        }
    }
}

//#Preview {
//    ContentView()
//}
