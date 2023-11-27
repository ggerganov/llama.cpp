// extended list
export const promptFormats = {  
"airoborosl2": {
  template: "{{prompt}} {{history}} {{char}}",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"alpaca": {
  template: "{{prompt}}\n\n{{history}}\n\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Response",
  user: "Instruction"
},
"chatml": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"codeCherryPop": {
  template: "{{prompt}}\n\n{{history}}\n\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Response",
  user: "Instruction"
},
"deepseekCoder": {
  template: "{{prompt}}\n{{history}}\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Response",
  user: "Instruction"
},
"dolphinMistral": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"evolvedSeeker": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"goliath120b": {
  template: "{{prompt}}\n\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"jordan": {
  template: "{{prompt}}\n\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"leoHessianai": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"leoMistral": {
  template: "{{prompt}} {{history}} {{char}}",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"llama2": {
  template: "<s>[INST] <<SYS>>\n{{prompt}}\n<</SYS>>\n\n{{history}} [/INST] {{char}} </s><s>[INST] ",
  historyTemplate: "{{name}}: {{message}} [/INST]",
  char: "llama",
  user: "user"
},
"marx": {
  template: "{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"med42": {
  template: "<|system|>: {{prompt}}\n{{history}}\n{{char}}",
  historyTemplate: "<|{{name}}|>:{{message}}",
  char: "assistant",
  user: "prompter"
},
"metaMath": {
  template: "{{prompt}}\n\n{{history}}\n\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Response",
  user: "Instruction"
},
"mistralInstruct": {
  template: "<s>{{history}} [/INST]\n{{char}}</s>",
  historyTemplate: "{{name}} {{message}}",
  char: "",
  user: "[INST] "
},
"mistralOpenOrca": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"mythomax": {
  template: "{{prompt}}\n\n{{history}}\n\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Response",
  user: "Instruction"
},
"neuralchat": {
  template: "### System:\n{{prompt}}\n{{history}}\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Assistant",
  user: "User"
},
"nousCapybara": {
  template: "{{history}}\n{{char}}",
  historyTemplate: "\n{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"nousHermes": {
  template: "### Instruction: {{prompt}}\n{{history}}\n### {{char}}:",
  historyTemplate: "\n### {{name}}: {{message}}",
  char: "Response",
  user: "Input"
},
"openhermes2Mistral": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"orcamini": {
  template: "{{prompt}}\n\n{{history}}\n\n### {{char}}:",
  historyTemplate: "### {{name}}:\n{{message}}",
  char: "Response",
  user: "Instruction"
},
"sauerkraut": {
  template: "{{prompt}}\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "Assistant",
  user: "User"
},
"samantha": {
  template: "{{prompt}}\n\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"samanthaMistral": {
  template: "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}\n<|im_start|>{{char}}",
  historyTemplate: "<|im_start|>{{user}}\n{{message}}<|im_end|>",
  char: "assistant",
  user: "user"
},
"scarlett": {
  template: "{{prompt}}\n\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"sydney": {
  template: "{{prompt}}\n\n{{history}}\n{{char}}",
  historyTemplate: "### {{name}}:\n{{message}}\n",
  char: "Response",
  user: "Instruction"
},
"synthia": {
  template: "SYSTEM: {{prompt}}\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"tess": {
  template: "SYSTEM: {{prompt}}\n{{history}}\n{{char}}:",
  historyTemplate: "{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"vicuna": {
  template: "{{prompt}}\n{{history}}\n{{char}}:",
  historyTemplate: "\n{{name}}: {{message}}",
  char: "ASSISTANT",
  user: "USER"
},
"yi34b": {
  template: "{{history}} {{char}}",
  historyTemplate: "{{name}}: {{message}}",
  char: "Assistant",
  user: "Human"
},
"zephyr": {
  template: "<|system|>\n{{prompt}}</s>\n{{history}}\n{{char}}",
  historyTemplate: "<|{{name}}|>\n{{message}}</s>",
  char: "assistant",
  user: "user"
}
};
