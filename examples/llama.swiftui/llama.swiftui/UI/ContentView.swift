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
                .padding(8)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)

                Button("Bench") {
                    bench()
                }
                .padding(8)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)

                Button("Clear") {
                    clear()
                }
                .padding(8)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)

                Button("Copy") {
                    UIPasteboard.general.string = llamaState.messageLog
                }
                .padding(8)
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }

            VStack {
                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TinyLlama-1.1B (Q4_0)",
                    modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",
                    filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf"
                )
                .font(.system(size: 12))
                .padding(.top, 4)

                DownloadButton(
                    llamaState: llamaState,
                    modelName: "TinyLlama-1.1B (Q8_0)",
                    modelUrl: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q8_0.gguf?download=true",
                    filename: "tinyllama-1.1b-1t-openorca.Q8_0.gguf"
                )
                .font(.system(size: 12))

                Button("Clear downloaded models") {
                    ContentView.cleanupModelCaches()
                    llamaState.cacheCleared = true
                }
                .padding(8)
                .font(.system(size: 12))
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

    func clear() {
        Task {
            await llamaState.clear()
        }
    }
}

//#Preview {
//    ContentView()
//}
