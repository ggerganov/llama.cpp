import Foundation

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false

    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }
    }

    func loadModel(modelUrl: URL?) throws {
        messageLog += "Loading model...\n"
        if let modelUrl {
            llamaContext = try LlamaContext.create_context(path: modelUrl.path())
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

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"
        messageLog += "Running benchmark...\n"
        await llamaContext.bench() // heat up
        let result = await llamaContext.bench()
        messageLog += "\(result)"
    }
}
