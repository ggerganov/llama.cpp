// extended list
export const promptFormats = {
"alpaca": {
template: `{{prompt}}\n\n{{history}}\n\n{{char}}:`,

historyTemplate: `### {{name}}:\n{{message}}`,

char: "Response",
charMsgPrefix: "",
charMsgSuffix: "",

user: "Instruction",
userMsgPrefix: "",
userMsgSuffix: "",

stops: ""
},

// ----------------------------

"chatml": {
template: `<|im_start|>system\n{{prompt}}<|im_end|>{{history}}{{char}}`,

historyTemplate: `\n<|im_start|>{{name}}\n{{message}}<|im_end|>`,

char: "assistant",
charMsgPrefix: "",
charMsgSuffix: "",

user: "user",
userMsgPrefix: "",
userMsgSuffix: "",

stops: ""
},

// ----------------------------

"llama2": {
template: `<s>[INST] <<SYS>>\n{{prompt}}\n<</SYS>>\n\nTest Message [/INST] Test Successfull </s>{{history}}{{char}}`,
historyTemplate: `{{name}}: {{message}}`,

char: "Assistant",
charMsgPrefix: "",
charMsgSuffix: "</s>",

user: "User",
userMsgPrefix: "<s>[INST] ",
userMsgSuffix: " [/INST]",

stops: ""
},
// ref: https://huggingface.co/blog/llama2#how-to-prompt-llama-2

// ----------------------------

"llama3": {
template: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{prompt}}{{history}}{{char}}`,
historyTemplate: `<|start_header_id|>{{name}}<|end_header_id|>\n\n{{message}}<|eot_id|>`,

char: "assistant",
charMsgPrefix: "",
charMsgSuffix: "",

user: "user",
userMsgPrefix: "",
userMsgSuffix: "",

stops: "<|eot_id|>"
},

// ----------------------------

"openchat": {
template: `{{history}}{{char}}`,
historyTemplate: `GPT4 Correct {{name}}: {{message}}<|end_of_turn|>`,

char: "Assistant",
charMsgPrefix: "",
charMsgSuffix: "",

user: "User",
userMsgPrefix: "",
userMsgSuffix: "",

stops: ""
},

// ----------------------------

"phi3": {
template: `{{history}}{{char}}`,

historyTemplate: `<|{{name}}|>\n{{message}}<|end|>\n`,

char: "assistant",
charMsgPrefix: "",
charMsgSuffix: "",

user: "user",
userMsgPrefix: "",
userMsgSuffix: "",

stops: "<|end|>"
},

// ----------------------------

"vicuna": {
template: `SYSTEM: {{prompt}}\n{{history}}{{char}}`,

historyTemplate: `
{{name}}: {{message}}\n`,

char: "ASSISTANT",
charMsgPrefix: "",
charMsgSuffix: "",

user: "USER",
userMsgPrefix: "",
userMsgSuffix: "",

stops: ""
},

// ----------------------------

"codeCherryPop": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},

// ----------------------------

"deepseekCoder": {
template: `{{prompt}}{{history}}{{char}}:`,

historyTemplate: `### {{name}}:\n{{message}}`,

char: "Response",
charMsgPrefix: "",
charMsgSuffix: "",

user: "Instruction",
userMsgPrefix: "",
userMsgSuffix: "",

stops: "<|EOT|>"
},

// ----------------------------

"goliath120b": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"jordan": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"llava": {
template: `{{history}}{{char}}:`,
historyTemplate: `{{name}}: {{message}}
`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"leoHessianai": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
<|im_start|>{{char}}`,
historyTemplate: `<|im_start|>{{user}}
{{message}}<|im_end|>`,
char: "assistant",
user: "user"
},

// ----------------------------

"leoMistral": {
template: `{{prompt}} {{history}} {{char}}`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"marx": {
template: `{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"med42": {
template: `<|system|>: {{prompt}}
{{history}}
{{char}}`,
historyTemplate: `<|{{name}}|>:{{message}}`,
char: "assistant",
user: "prompter"
},

// ----------------------------

"metaMath": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},

// ----------------------------

"mistralInstruct": {
template: `<s>[INST] ({{prompt}}) {{history}} {{char}}</s>`,
historyTemplate: `{{name}} {{message}}`,
char: "[/INST] Assistant:",
user: "[INST] User:"
},

// ----------------------------

"mistralOpenOrca": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
<|im_start|>{{char}}`,
historyTemplate: `<|im_start|>{{user}}
{{message}}<|im_end|>`,
char: "assistant",
user: "user"
},

// ----------------------------

"mythomax": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},

// ----------------------------

"neuralchat": {
template: `### System:
{{prompt}}
{{history}}
### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Assistant",
user: "User"
},

// ----------------------------

"nousCapybara": {
template: `{{history}}
{{char}}`,
historyTemplate: `
{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"nousHermes": {
template: `### Instruction: {{prompt}}
{{history}}
### {{char}}:`,
historyTemplate: `
### {{name}}: {{message}}`,
char: "Response",
user: "Input"
},

// ----------------------------

"openchatMath": {
template: `{{history}}{{char}}`,
historyTemplate: `Math Correct {{name}}: {{message}}<|end_of_turn|>`,
char: "Assistant",
user: "User"
},

// ----------------------------

"openhermes2Mistral": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
<|im_start|>{{char}}`,
historyTemplate: `<|im_start|>{{user}}
{{message}}<|im_end|>`,
char: "assistant",
user: "user"
},

// ----------------------------

"orcamini": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},

// ----------------------------

"orion": {
template: `<s>Human: Test Message\n\nAssistant: </s>Test Successful</s>{{history}}{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,

char: "Assistant </s>",
charMsgPrefix: "",
charMsgSuffix: "",

user: "Human",
userMsgPrefix: "",
userMsgSuffix: "\n\n",

stops: ""
},

// ----------------------------

"sauerkraut": {
template: `{{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "Assistant",
user: "User"
},

// ----------------------------

"samantha": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"samanthaMistral": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
<|im_start|>{{char}}`,
historyTemplate: `<|im_start|>{{user}}
{{message}}<|im_end|>`,
char: "assistant",
user: "user"
},

// ----------------------------

"scarlett": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"starlingCode": {
template: `{{history}}{{char}}`,
historyTemplate: `Code {{name}}: {{message}}<|end_of_turn|>`,
char: "Assistant",
user: "User"
},

// ----------------------------

"sydney": {
template: `{{prompt}}

{{history}}
{{char}}`,
historyTemplate: `### {{name}}:
{{message}}
`,
char: "Response",
user: "Instruction"
},

// ----------------------------

"synthia": {
template: `SYSTEM: {{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"tess": {
template: `SYSTEM: {{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},

// ----------------------------

"yi34b": {
template: `{{history}} {{char}}`,
historyTemplate: `{{name}}: {{message}}`,
char: "Assistant",
user: "Human"
},

// ----------------------------

"zephyr": {
template: `<|system|>
{{prompt}}</s>
{{history}}
{{char}}`,
historyTemplate: `<|{{name}}|>
{{message}}</s>`,
char: "assistant",
user: "user"
}
};
