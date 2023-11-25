import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()

    @State private var multiLineText = ""

    var body: some View {
        VStack {
            ScrollView(.vertical) {
                Text(llamaState.messageLog)
            }

            TextEditor(text: $multiLineText)
                .frame(height: 200)
                .padding()
                .border(Color.gray, width: 0.5)
            Button(action: {
                sendText()
            }) {
                Text("Send")
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
}
/*
#Preview {
    ContentView()
}
*/
