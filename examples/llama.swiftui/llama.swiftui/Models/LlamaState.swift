import Foundation

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""

    private var llamaContext: LlamaContext?
    private var modelUrl: URL? {
        Bundle.main.url(forResource: "q8_0", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }
    init() {
        do {
            try loadModel()
        } catch {
            messageLog += "Error!\n"
        }
    }

    private func loadModel() throws {
        messageLog += "Loading model...\n"
        if let modelUrl {
            llamaContext = try LlamaContext.createContext(path: modelUrl.path())
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
        } else {
            messageLog += "Could not locate model\n"
        }
    }

    func complete(text: String) async {
        guard let llamaContext else {
            return
        }
        messageLog += "Attempting to complete text...\n"
        await llamaContext.completion_init(text: text)
        messageLog += "\(text)"

        while await llamaContext.n_cur <= llamaContext.n_len {
            let result = await llamaContext.completion_loop()
            messageLog += "\(result)"
        }
        await llamaContext.clear()
        messageLog += "\n\ndone\n"
    }
}
