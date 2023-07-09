const paramDefaults = {
  stream: true,
  n_predict: 500,
  temperature: 0.2,
  stop: ["</s>"]
};

let generation_settings = null;


// Completes the prompt as a generator. Recommended for most use cases.
//
// Example:
//
//    import { llama } from '/completion.js'
//
//    const request = llama("Tell me a joke", {n_predict: 800})
//    for await (const chunk of request) {
//      document.write(chunk.data.content)
//    }
//
export async function* llama(prompt, params = {}, config = {}) {
  let controller = config.controller;

  if (!controller) {
    controller = new AbortController();
  }

  const completionParams = { ...paramDefaults, ...params, prompt };

  const response = await fetch("/completion", {
    method: 'POST',
    body: JSON.stringify(completionParams),
    headers: {
      'Connection': 'keep-alive',
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream'
    },
    signal: controller.signal,
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let content = "";

  try {
    let cont = true;

    while (cont) {
      const result = await reader.read();
      if (result.done) {
        break;
      }

      // sse answers in the form multiple lines of: value\n with data always present as a key. in our case we
      // mainly care about the data: key here, which we expect as json
      const text = decoder.decode(result.value);

      // parse all sse events and add them to result
      const regex = /^(\S+):\s(.*)$/gm;
      for (const match of text.matchAll(regex)) {
        result[match[1]] = match[2]
      }

      // since we know this is llama.cpp, let's just decode the json in data
      result.data = JSON.parse(result.data);
      content += result.data.content;

      // yield
      yield result;

      // if we got a stop token from server, we will break here
      if (result.data.stop) {
        if (result.data.generation_settings) {
          generation_settings = result.data.generation_settings;
        }
        break;
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      console.error("llama error: ", e);
    }
    throw e;
  }
  finally {
    controller.abort();
  }

  return content;
}

// Call llama, return an event target that you can subcribe to
//
// Example:
//
//    import { llamaEventTarget } from '/completion.js'
//
//    const conn = llamaEventTarget(prompt)
//    conn.addEventListener("message", (chunk) => {
//      document.write(chunk.detail.content)
//    })
//
export const llamaEventTarget = (prompt, params = {}, config = {}) => {
  const eventTarget = new EventTarget();
  (async () => {
    let content = "";
    for await (const chunk of llama(prompt, params, config)) {
      if (chunk.data) {
        content += chunk.data.content;
        eventTarget.dispatchEvent(new CustomEvent("message", { detail: chunk.data }));
      }
      if (chunk.data.generation_settings) {
        eventTarget.dispatchEvent(new CustomEvent("generation_settings", { detail: chunk.data.generation_settings }));
      }
      if (chunk.data.timings) {
        eventTarget.dispatchEvent(new CustomEvent("timings", { detail: chunk.data.timings }));
      }
    }
    eventTarget.dispatchEvent(new CustomEvent("done", { detail: { content } }));
  })();
  return eventTarget;
}

// Call llama, return a promise that resolves to the completed text. This does not support streaming
//
// Example:
//
//     llamaPromise(prompt).then((content) => {
//       document.write(content)
//     })
//
//     or
//
//     const content = await llamaPromise(prompt)
//     document.write(content)
//
export const llamaPromise = (prompt, params = {}, config = {}) => {
  return new Promise(async (resolve, reject) => {
    let content = "";
    try {
      for await (const chunk of llama(prompt, params, config)) {
        content += chunk.data.content;
      }
      resolve(content);
    } catch (error) {
      reject(error);
    }
  });
};

/**
 * (deprecated)
 */
export const llamaComplete = async (params, controller, callback) => {
  for await (const chunk of llama(params.prompt, params, { controller })) {
    callback(chunk);
  }
}

// Get the model info from the server. This is useful for getting the context window and so on.
export const llamaModelInfo = async () => {
  if (!generation_settings) {
    generation_settings = await fetch("/model.json").then(r => r.json());
  }
  return generation_settings;
}
