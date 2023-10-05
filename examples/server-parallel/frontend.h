const char* system_prompt_default =
R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
User: Recommend a nice restaurant in the area.
Assistant: I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User: Who is Richard Feynman?
Assistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
User:)";

const char* index_html_ = R"(
<!DOCTYPE html>
<html>
<head>
  <title>llama.cpp - server parallel PoC</title>
  <script src="index.js"></script>
</head>
<body>
  <div style="width: 90%;margin: auto;">
    <h2>Server parallel - PoC</h2>
    <form id="myForm">
      <input type="checkbox" id="system_promt_cb" name="myCheckbox" onchange="toggleSP() ">
      <label for="system_promt_cb">Use custom system prompt</label>
      <br>
      <div id="system_prompt_view" style="display: none;">
        <textarea id="sp_text" name="systemPrompt" style="width: 100%;height: 4rem;" placeholder="System Prompt"></textarea>
        <label for="user_name">User name</label>
        <input type="text" id="user_name" value=""  placeholder="Anti prompt" required>
        <label for="assistant_name">Assistant name</label>
        <input type="text" id="assistant_name" value="" placeholder="Assistant:" required>
        <button type="button" id="btn_reset" onclick="clearSP() " >Clear all</button>
      </div>
      <br>
      <label for="slot_id">Slot ID (-1 load in a idle slot)</label>
      <input type="number" id="slot_id" value="-1" required>
      <br>
      <label for="temperature">Temperature</label>
      <input type="number" id="temperature" value="0.1" required>
      <br>
      <label for="message">Message</label>
      <input id="message" style="width: 80%;" required>
      <br><br>
      <button type="button" id="btn_send" onclick="perform() " >Send</button>
      <br>
      <br>
      <button type="button" id="btn_reset" onclick="resetBtn() " >Reset</button>
    </form>
    <div id="conversation_view">
    </div>
  </div>
</body>
</html>
)";

const char* index_js_ = R"(
 let conversation = [];
 let current_message = -1;

const questions = [
  "Who is Elon Musk?",
  "Who is Jeff Bezos?",
  "How to get a job at google?",
  "What are you?",
  "When was born Abraham Lincoln?",
];

let user_name = "";
let assistant_name = "";

function toggleSP() {
    if(document.getElementById("system_promt_cb").checked) {
        document.getElementById("system_prompt_view").style.display = "block";
    } else {
        document.getElementById("system_prompt_view").style.display = "none";
    }
}

function clearSP() {
    document.getElementById("sp_text").value = "";
    document.getElementById("anti_prompt").value = "";
    document.getElementById("assistant_name").value = "";
}

docReady(async () => {
  document.getElementById("message").value =
    questions[Math.floor(Math.random() * questions.length)];

    // to keep the same prompt format in all clients
    const response = await fetch("/props");
    if (!response.ok) {
      alert(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    user_name = data.user_name;
    assistant_name = data.assistant_name;
});

function docReady(fn) {
  // see if DOM is already available
  if (
    document.readyState === "complete" ||
    document.readyState === "interactive"
  ) {
    // call on next available tick
    setTimeout(fn, 1);
  } else {
    document.addEventListener("DOMContentLoaded", fn);
  }
}

function updateView() {
  let conv_view = document.getElementById("conversation_view");
  // build view
  conv_view.innerHTML = "";
  for (let index in conversation) {
    conversation[index].assistant = conversation[index].assistant.replace(
        user_name,
      ""
    );
    conv_view.innerHTML += `
          <p><span style="font-weight: bold">User:</span> ${conversation[index].user}<p>
          <p style="white-space: pre-line;"><span style="font-weight: bold">Assistant:</span> ${conversation[index].assistant}<p>`;
  }
}

async function call_llama(options) {
  const response = await fetch("/completion", {
    method: "POST",
    body: JSON.stringify(options),
    headers: {
      Connection: "keep-alive",
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
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
        document.getElementById("btn_send").disabled = false;
        break;
      }

      // Add any leftover data to the current chunk of data
      const text = leftover + decoder.decode(result.value);

      // Check if the last character is a line break
      const endsWithLineBreak = text.endsWith("\n");

      // Split the text into lines
      let lines = text.split("\n");

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
          result[match[1]] = match[2];
          // since we know this is llama.cpp, let's just decode the json in data
          if (result.data) {
            result.data = JSON.parse(result.data);
            conversation[current_message].assistant += result.data.content;
            updateView();
          }
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      console.error("llama error: ", e);
    }
    throw e;
  }
}

function generatePrompt() {
  // generate a good prompt to have coherence
  let prompt = "";
  for (let index in conversation) {
    if (index == 0) {
      prompt += conversation[index].user + "\n";
    } else {
      prompt += user_name + conversation[index].user + "\n";
    }
    if (index == current_message) {
      prompt += assistant_name;
    } else {
      prompt += assistant_name + conversation[index].assistant;
    }
  }
  return prompt;
}

function resetBtn() {
  document.getElementById("slot_id").value = "-1";
  document.getElementById("temperature").value = "0.1";
  document.getElementById("message").value =
    questions[Math.floor(Math.random() * questions.length)];
  document.getElementById("conversation_view").innerHTML = "";
  conversation = [];
  current_message = -1;
}

async function perform() {
  var slot_id = parseInt(document.getElementById("slot_id").value);
  var temperature = parseFloat(document.getElementById("temperature").value);
  var prompt = " " + document.getElementById("message").value;
  if (!isNaN(slot_id) && !isNaN(temperature) && prompt.length > 0) {
    let options = {
        slot_id,
        temperature
    };
    if(document.getElementById("system_promt_cb").checked) {
        let system_prompt = document.getElementById("sp_text").value;
        let anti_prompt = document.getElementById("user_name").value;
        let assistant_name_ = document.getElementById("assistant_name").value;
        if(!system_prompt || !anti_prompt || !assistant_name_) {
          document.getElementById("conversation_view").innerText =
                "please, insert valid props.";
          return;
        }
        conversation = [];
        current_message = -1;
        document.getElementById("system_promt_cb").checked = false;
        document.getElementById("system_promt_cb").dispatchEvent(new Event("change"));
        options.system_prompt = system_prompt;
        options.anti_prompt = anti_prompt;
        options.assistant_name = assistant_name_;
        user_name = anti_prompt;
        assistant_name = assistant_name_;
    }
    current_message++;
    conversation.push({
      user: prompt,
      assistant: "",
    });
    updateView();
    document.getElementById("message").value = "";
    document.getElementById("btn_send").disabled = true;
    options.prompt = generatePrompt();
    await call_llama(options);
  } else {
    document.getElementById("conversation_view").innerText =
      "please, insert valid props.";
  }
}

)";
