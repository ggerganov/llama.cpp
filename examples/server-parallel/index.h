
const auto index_html = R"(
   <!DOCTYPE html>
<html>
<head>
  <title>llama.cpp - server parallel PoC</title>
</head>
<body>
  <div style="width: 90%;margin: auto;">
    <h2>Server parallel - Proof of Concept</h2>
    <form id="myForm">
      <label for="client_slot">Client Slot (-1 load in a idle client)</label>
      <input type="number" id="client_slot" value="-1" required>
      <br><br>
      <label for="message">Message</label>
      <input id="message" style="width: 80%;" required>
      <br><br>
      <button type="button" id="btn_send" onclick="perform() " >Send</button>
      <button type="button" onclick="reset() " >Reset</button>
    </form>
    <div id="conversation_view">
    </div>
  </div>

  <script>
    let conversation = [];
    let current_message = -1;
    const questions = ["Who is Elon Musk?", "Who is Jeff Bezos?", "How to get a job at google?", "What are you?", "When was born Abraham Lincoln?"];
    window.onload = function() {
      document.getElementById("message").value = questions[Math.floor(Math.random() * questions.length)];
    };

    function updateView() {
      let conv_view = document.getElementById("conversation_view");
      // build view
      conv_view.innerHTML = "";
      for(let index in conversation) {
        conversation[index].assistant = conversation[index].assistant.replace("User:", "");
        conv_view.innerHTML += `
          <p><span style="font-weight: bold">User:</span> ${conversation[index].user}<p>
          <p style="white-space: pre-line;"><span style="font-weight: bold">Assistant:</span> ${conversation[index].assistant}<p>`;
      }
    }

    async function call_llama(options) {
      const response = await fetch("/completion", {
        method: 'POST',
        body: JSON.stringify(options),
        headers: {
          'Connection': 'keep-alive',
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        }
      });

      const reader = response.body.getReader();
      let cont = true;
      const decoder = new TextDecoder();
      let leftover = ""; // Buffer for partially read lines

      try {
        let cont = true;

        while (cont) {
          const result = await reader.read();
          if (result.done) {
            break;
          }

          // Add any leftover data to the current chunk of data
          const text = leftover + decoder.decode(result.value);

          // Check if the last character is a line break
          const endsWithLineBreak = text.endsWith('\n');

          // Split the text into lines
          let lines = text.split('\n');

          // If the text doesn't end with a line break, then the last line is incomplete
          // Store it in leftover to be added to the next chunk of data
          if (!endsWithLineBreak) {
            leftover = lines.pop();
          } else {
            leftover = ""; // Reset leftover if we have a line break at the end
          }

          // Parse all sse events and add them to result
          const regex = /^(\S+):\s(.*)$/gm;
          for (const line of lines) {
            const match = regex.exec(line);
            if (match) {
              result[match[1]] = match[2]
              // since we know this is llama.cpp, let's just decode the json in data
              if (result.data) {
                result.data = JSON.parse(result.data);
                conversation[current_message].assistant += result.data.token;
                updateView();
              }
            }
          }
        }
      } catch (e) {
        if (e.name !== 'AbortError') {
          console.error("llama error: ", e);
        }
        throw e;
      }
    }

    function generatePrompt() {
      let prompt = '';
      for(let index in conversation) {
        if(index == 0) {
          prompt += conversation[index].user + "\n";
        } else {
          prompt += "User: " + conversation[index].user + "\n";
        }
        if(index == current_message) {
          prompt += "Assistant:";
        } else {
          prompt += "Assistant: " + conversation[index].assistant;
        }
      }
      return prompt;
    }

    function reset() {
      conversation = [];
      document.getElementById("client_slot").value = "-1";
      document.getElementById("message").value = "";
      document.getElementById("conversation_view").innerHTML = "";
    }

    async function perform() {
      var client_slot = parseFloat(document.getElementById("client_slot").value);
      var prompt = document.getElementById("message").value;
      if (!isNaN(client_slot) && prompt.length > 0) {
        current_message++;
        conversation.push({
          user: prompt,
          assistant: ''
        });
        updateView();
        await call_llama({
          client_slot,
          prompt: generatePrompt()
        });
      } else {
        document.getElementById("conversation_view").innerText = "please, insert valid props.";
      }
    }
  </script>
</body>
</html>
)";
