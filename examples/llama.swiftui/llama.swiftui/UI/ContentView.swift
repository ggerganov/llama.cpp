import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()

    @State private var multiLineText = ""

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

#Preview {
    ContentView()
}
