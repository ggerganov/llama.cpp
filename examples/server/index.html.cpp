// Generated file, run deps.sh to update. Do not edit directly
R"htmlraw(<html>

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>llama.cpp - chat</title>

  <style>
    #container {
      max-width: 80rem;
      margin: 4em auto;
    }

    main {
      border: 1px solid #ddd;
      padding: 1em;
    }

    #chat {
      height: 50vh;
      overflow-y: auto;
    }

    body {
      max-width: 650px;
      line-height: 1.2;
      font-size: 16px;
      margin: 0 auto;
    }

    p {
      overflow-wrap: break-word;
      word-wrap: break-word;
      hyphens: auto;
      margin-top: 0.5em;
      margin-bottom: 0.5em;
    }

    form {
      margin: 1em 0 0 0;
      display: flex;
      gap: 0.5em;
      flex-direction: row;
      align-items: center;
    }

    form > * {
      padding: 4px;
    }

    form input {
      flex-grow: 1;
    }

    fieldset {
      width: 100%;
      padding: 1em;
    }

    fieldset label {
      margin: 0.5em 0;
      display: block;
    }
  </style>


  <script type="module">
    import {
      html, h, signal, effect, computed, render, useSignal, useEffect, useRef, fetchEventSource
    } from '/index.js';

    const transcript = signal([])
    const chatStarted = computed(() => transcript.value.length > 0)

    const chatTemplate = signal("{{prompt}}\n\n{{history}}\n{{bot}}:")
    const settings = signal({
      prompt: "This is a conversation between user and llama, a friendly chatbot.",
      bot: "llama",
      user: "User"
    })

    const temperature = signal(0.2)
    const nPredict = signal(80)
    const controller = signal(null)
    const generating = computed(() => controller.value == null )

    // simple template replace
    const template = (str, map) => {
      let params = settings.value;
      if (map) {
        params = { ...params, ...map };
      }
      return String(str).replaceAll(/\{\{(.*?)\}\}/g, (_, key) => template(params[key]));
    }

    // send message to server
    const chat = async (msg) => {
      if (controller.value) {
        console.log('already running...');
        return;
      }
      controller.value = new AbortController();

      const history = [...transcript.value, ['{{user}}', msg]];
      transcript.value = history;

      let additionalParams = {
        message: msg,
        history: history.flatMap(([name, msg]) => `${name}: ${msg}`).join("\n"),
      }

      const payload = template(chatTemplate.value, additionalParams)

      let currentMessage = "";
      await fetchEventSource('/completion', {
        method: 'POST',
        signal: controller.value.signal,
        body: JSON.stringify({
          stream: true,
          prompt: payload,
          n_predict: parseInt(nPredict.value),
          temperature: parseFloat(temperature.value),
          stop: ["</s>", template("{{bot}}:"), template("{{user}}:")]
        }),
        onmessage(e) {
          const data = JSON.parse(e.data);
          currentMessage += data.content;

          if (data.stop) {
            console.log("-->", data, ' response was:', currentMessage, 'transcript state:', transcript.value);
          }

          transcript.value = [...history, ['{{bot}}', currentMessage]]
          return true;
        },
        onclose(e) {
          controller.value = null;
          return false;
        },
      });
    }

    function MessageInput() {
      const message = useSignal("")

      const stop = (e) => {
        e.preventDefault();
        if (controller.value) {
          controller.value.abort();
          controller.value = null;
        }
      }

      const reset = (e) => {
        stop(e);
        transcript.value = [];
      }

      const submit = (e) => {
        stop(e);
        chat(message.value);
        message.value = "";
      }

      return html`
        <form onsubmit=${submit}>
          <input type="text" value="${message}" oninput=${(e) => message.value = e.target.value} autofocus placeholder="Chat here..."/>
          <button type="submit" disabled=${!generating.value} >Send</button>
          <button onclick=${stop} disabled=${generating}>Stop</button>
          <button onclick=${reset}>Reset</button>
        </form>
      `
    }

    const ChatLog = (props) => {
      const messages = transcript.value;
      const container = useRef(null)

      useEffect(() => {
        // scroll to bottom (if needed)
        if (container.current && container.current.scrollHeight <= container.current.scrollTop + container.current.offsetHeight + 100) {
          container.current.scrollTo(0, container.current.scrollHeight)
        }
      }, [messages])

      const chatLine = ([user, msg]) => {
        return html`<p key=${msg}><strong>${template(user)}:</strong> ${template(msg)}</p>`
      };

      return html`
        <section id="chat" ref=${container}>
          ${messages.flatMap((m) => chatLine(m))}
        </section>`;
    };

    const ConfigForm = (props) => {

      return html`
        <form>
          <fieldset>
            <legend>Settings</legend>

            <div>
              <label for="prompt">Prompt</label>
              <textarea type="text" id="prompt" value="${settings.value.prompt}" oninput=${(e) => settings.value.prompt = e.target.value} rows="3" cols="60" />
            </div>

            <div>
              <label for="user">User name</label>
              <input type="text" id="user" value="${settings.value.user}" oninput=${(e) => settings.value.user = e.target.value} />
            </div>

            <div>
              <label for="bot">Bot name</label>
              <input type="text" id="bot" value="${settings.value.bot}" oninput=${(e) => settings.value.bot = e.target.value} />
            </div>

            <div>
              <label for="template">Prompt template</label>
              <textarea id="template" value="${chatTemplate}" oninput=${(e) => chatTemplate.value = e.target.value} rows="8" cols="60" />
            </div>

            <div>
              <label for="temperature">Temperature</label>
              <input type="range" id="temperature" min="0.0" max="1.0" step="0.01" value="${temperature.value}" oninput=${(e) => temperature.value = e.target.value} />
              <span>${temperature}</span>
            </div>

            <div>
              <label for="nPredict">Predictions</label>
              <input type="range" id="nPredict" min="1" max="2048" step="1" value="${nPredict.value}" oninput=${(e) => nPredict.value = e.target.value} />
              <span>${nPredict}</span>
            </div>
            </fieldset>

        </form>
      `

    }

    function App(props) {

      return html`
      <div id="container">
        <header>
          <h1>llama.cpp</h1>
        </header>

        <main>
          <section class="chat">
            <${chatStarted.value ? ChatLog : ConfigForm
        } />
          </section>

          <hr/>

          <section class="chat">
            <${MessageInput} />
          </section>
        </main>

        <footer>
          <p>Powered by <a href="https://github.com/ggerganov/llama.cpp">llama.cpp</a> and <a href="https://ggml.ai">ggml.ai</a></p>
        </footer>
      </div>
      `;
    }

    render(h(App), document.body);
  </script>
</head>

<body>
</body>

</html>)htmlraw"

