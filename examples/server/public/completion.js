const paramDefaults = {
  stream: true,
  n_predict: 500,
  temperature: 0.2,
  stop: ["</s>"]
};

/**
 * This function completes the input text using a llama dictionary.
 * @param {object} params - The parameters for the completion request.
 * @param {object} controller - an instance of AbortController if you need one, or null.
 * @param {function} callback - The callback function to call when the completion is done.
 * @returns {string} the completed text as a string. Ideally ignored, and you get at it via the callback.
 */
export const llamaComplete = async (params, controller, callback) => {
  if (!controller) {
    controller = new AbortController();
  }
  const completionParams = { ...paramDefaults, ...params };

  // we use fetch directly here becasue the built in fetchEventSource does not support POST
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

      // callack
      if (callback) {
        cont = callback(result) != false;
      }

      // if we got a stop token from server, we will break here
      if (result.data.stop) {
        break;
      }
    }
  } catch (e) {
    console.error("llama error: ", e);
    throw e;
  }
  finally {
    controller.abort();
  }

  return content;
}
