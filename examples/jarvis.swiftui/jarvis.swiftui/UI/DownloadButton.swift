import SwiftUI

struct DownloadButton: View {
    @ObservedObject private var jarvisState: JarvisState
    private var modelName: String
    private var modelUrl: String
    private var filename: String

    @State private var status: String

    @State private var downloadTask: URLSessionDownloadTask?
    @State private var progress = 0.0
    @State private var observation: NSKeyValueObservation?

    private static func getFileURL(filename: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
    }

    private func checkFileExistenceAndUpdateStatus() {
    }

    init(jarvisState: JarvisState, modelName: String, modelUrl: String, filename: String) {
        self.jarvisState = jarvisState
        self.modelName = modelName
        self.modelUrl = modelUrl
        self.filename = filename

        let fileURL = DownloadButton.getFileURL(filename: filename)
        status = FileManager.default.fileExists(atPath: fileURL.path) ? "downloaded" : "download"
    }

    private func download() {
        status = "downloading"
        print("Downloading model \(modelName) from \(modelUrl)")
        guard let url = URL(string: modelUrl) else { return }
        let fileURL = DownloadButton.getFileURL(filename: filename)

        downloadTask = URLSession.shared.downloadTask(with: url) { temporaryURL, response, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }

            guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
                print("Server error!")
                return
            }

            do {
                if let temporaryURL = temporaryURL {
                    try FileManager.default.copyItem(at: temporaryURL, to: fileURL)
                    print("Writing to \(filename) completed")

                    jarvisState.cacheCleared = false

                    let model = Model(name: modelName, url: modelUrl, filename: filename, status: "downloaded")
                    jarvisState.downloadedModels.append(model)
                    status = "downloaded"
                }
            } catch let err {
                print("Error: \(err.localizedDescription)")
            }
        }

        observation = downloadTask?.progress.observe(\.fractionCompleted) { progress, _ in
            self.progress = progress.fractionCompleted
        }

        downloadTask?.resume()
    }

    var body: some View {
        VStack {
            if status == "download" {
                Button(action: download) {
                    Text("Download " + modelName)
                }
            } else if status == "downloading" {
                Button(action: {
                    downloadTask?.cancel()
                    status = "download"
                }) {
                    Text("\(modelName) (Downloading \(Int(progress * 100))%)")
                }
            } else if status == "downloaded" {
                Button(action: {
                    let fileURL = DownloadButton.getFileURL(filename: filename)
                    if !FileManager.default.fileExists(atPath: fileURL.path) {
                        download()
                        return
                    }
                    do {
                        try jarvisState.loadModel(modelUrl: fileURL)
                    } catch let err {
                        print("Error: \(err.localizedDescription)")
                    }
                }) {
                    Text("Load \(modelName)")
                }
            } else {
                Text("Unknown status")
            }
        }
        .onDisappear() {
            downloadTask?.cancel()
        }
        .onChange(of: jarvisState.cacheCleared) { newValue in
            if newValue {
                downloadTask?.cancel()
                let fileURL = DownloadButton.getFileURL(filename: filename)
                status = FileManager.default.fileExists(atPath: fileURL.path) ? "downloaded" : "download"
            }
        }
    }
}

// #Preview {
//    DownloadButton(
//        jarvisState: JarvisState(),
//        modelName: "TheBloke / TinyJarvis-1.1B-1T-OpenOrca-GGUF (Q4_0)",
//        modelUrl: "https://huggingface.co/TheBloke/TinyJarvis-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyjarvis-1.1b-1t-openorca.Q4_0.gguf?download=true",
//        filename: "tinyjarvis-1.1b-1t-openorca.Q4_0.gguf"
//    )
// }
