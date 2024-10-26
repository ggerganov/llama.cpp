import SwiftUI

struct ContentView: View {
    @StateObject var jarvisState = JarvisState()
    @State private var multiLineText = ""
    @State private var showingHelp = false    // To track if Help Sheet should be shown

    var body: some View {
        NavigationView {
            VStack {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(jarvisState.messageLog)
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
                        UIPasteboard.general.string = jarvisState.messageLog
                    }
                }
                .buttonStyle(.bordered)
                .padding()

                NavigationLink(destination: DrawerView(jarvisState: jarvisState)) {
                    Text("View Models")
                }
                .padding()

            }
            .padding()
            .navigationBarTitle("Model Settings", displayMode: .inline)

        }
    }

    func sendText() {
        Task {
            await jarvisState.complete(text: multiLineText)
            multiLineText = ""
        }
    }

    func bench() {
        Task {
            await jarvisState.bench()
        }
    }

    func clear() {
        Task {
            await jarvisState.clear()
        }
    }
    struct DrawerView: View {

        @ObservedObject var jarvisState: JarvisState
        @State private var showingHelp = false
        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = jarvisState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                do {
                    try FileManager.default.removeItem(at: fileURL)
                } catch {
                    print("Error deleting file: \(error)")
                }
            }

            // Remove models from downloadedModels array
            jarvisState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            return paths[0]
        }
        var body: some View {
            List {
                Section(header: Text("Download Models From Hugging Face")) {
                    HStack {
                        InputButton(jarvisState: jarvisState)
                    }
                }
                Section(header: Text("Downloaded Models")) {
                    ForEach(jarvisState.downloadedModels) { model in
                        DownloadButton(jarvisState: jarvisState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                    .onDelete(perform: delete)
                }
                Section(header: Text("Default Models")) {
                    ForEach(jarvisState.undownloadedModels) { model in
                        DownloadButton(jarvisState: jarvisState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                }

            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Model Settings", displayMode: .inline).toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelp = true
                    }
                }
            }.sheet(isPresented: $showingHelp) {    // Sheet for help modal
                VStack(alignment: .leading) {
                    VStack(alignment: .leading) {
                        Text("1. Make sure the model is in GGUF Format")
                               .padding()
                        Text("2. Copy the download link of the quantized model")
                               .padding()
                    }
                    Spacer()
                   }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
