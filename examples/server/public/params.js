import { signal } from './index.js';

export const session = signal({
      prompt: "This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.",
      template: "{{prompt}}\n\n{{history}}\n{{char}}:",
      historyTemplate: "{{name}}: {{message}}",
      transcript: [],
      type: "chat",  // "chat" | "completion"
      char: "Llama",
      user: "User",
      image_selected: ''
})

export const params = signal({
      n_predict: 400,
      temperature: 0.7,
      repeat_last_n: 256, // 0 = disable penalty, -1 = context size
      repeat_penalty: 1.18, // 1.0 = disabled
      penalize_nl: false,
      top_k: 40, // <= 0 to use vocab size
      top_p: 0.95, // 1.0 = disabled
      min_p: 0.05, // 0 = disabled
      tfs_z: 1.0, // 1.0 = disabled
      typical_p: 1.0, // 1.0 = disabled
      presence_penalty: 0.0, // 0.0 = disabled
      frequency_penalty: 0.0, // 0.0 = disabled
      mirostat: 0, // 0/1/2
      mirostat_tau: 5, // target entropy
      mirostat_eta: 0.1, // learning rate
      grammar: '',
      n_probs: 0, // no completion_probabilities,
      min_keep: 0, // min probs from each sampler,
      image_data: [],
      cache_prompt: true,
      api_key: ''
})
