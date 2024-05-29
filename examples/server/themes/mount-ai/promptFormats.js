// extended list
export const promptFormats = {
"alpaca": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},



"chatml": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
{{char}}`,
historyTemplate: `{{name}}
{{message}}`,
char: "<|im_start|>assistant",
user: "<|im_start|>user",
userMsgSuffix: "<|im_end|>"
},



"llama2": {
template: `<s>[INST] <<SYS>>
{{prompt}}
<</SYS>>

{{history}} [/INST] {{char}} </s><s>[INST] `,
historyTemplate: `{{name}}: {{message}} [/INST]`,
char: "llama",
user: "user",
userMsgSuffix: ""
},



"llama3": {
template: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{prompt}}<|eot_id|>{{history}}`,
historyTemplate: `<|start_header_id|>{{name}}<|end_header_id|>

{{message}}`,
char: "assistant",
user: "user",
userMsgSuffix: "<|eot_id|>"
},



"phi3": {
template: `{{history}}
{{char}}
`,
historyTemplate: `{{name}}
{{message}}`,
char: "<|assistant|>",
user: "<|user|>",
userMsgSuffix: "<|end|>"
},



"airoborosl2": {
template: `{{prompt}} {{history}} {{char}}`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"bakllava": {
template: `{{history}}{{char}}:`,
historyTemplate: `{{name}}: {{message}}
`,
char: "ASSISTANT",
user: "USER"
},



"codeCherryPop": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},



"deepseekCoder": {
template: `{{prompt}}
{{history}}
### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},



"dolphinMistral": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
<|im_start|>{{char}}`,
historyTemplate: `<|im_start|>{{user}}
{{message}}<|im_end|>`,
char: "assistant",
user: "user"
},



"evolvedSeeker": {
template: `<|im_start|>system
{{prompt}}<|im_end|>
{{history}}
<|im_start|>{{char}}`,
historyTemplate: `<|im_start|>{{user}}
{{message}}<|im_end|>`,
char: "assistant",
user: "user"
},



"goliath120b": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"jordan": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"llava": {
template: `{{history}}{{char}}:`,
historyTemplate: `{{name}}: {{message}}
`,
char: "ASSISTANT",
user: "USER"
},



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



"leoMistral": {
template: `{{prompt}} {{history}} {{char}}`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"marx": {
template: `{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"med42": {
template: `<|system|>: {{prompt}}
{{history}}
{{char}}`,
historyTemplate: `<|{{name}}|>:{{message}}`,
char: "assistant",
user: "prompter"
},



"metaMath": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},



"mistralInstruct": {
template: `<s>[INST] ({{prompt}}) {{history}} {{char}}</s>`,
historyTemplate: `{{name}} {{message}}`,
char: "[/INST] Assistant:",
user: "[INST] User:"
},



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



"mythomax": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},



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



"nousCapybara": {
template: `{{history}}
{{char}}`,
historyTemplate: `
{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"nousHermes": {
template: `### Instruction: {{prompt}}
{{history}}
### {{char}}:`,
historyTemplate: `
### {{name}}: {{message}}`,
char: "Response",
user: "Input"
},



"openChat": {
template: `{{history}}{{char}}`,
historyTemplate: `GPT4 {{user}}: {{message}}<|end_of_turn|>`,
char: "Assistant",
user: "User"
},



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



"orcamini": {
template: `{{prompt}}

{{history}}

### {{char}}:`,
historyTemplate: `### {{name}}:
{{message}}`,
char: "Response",
user: "Instruction"
},



"sauerkraut": {
template: `{{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "Assistant",
user: "User"
},



"samantha": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



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



"scarlett": {
template: `{{prompt}}

{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"starlingLM": {
template: `{{history}}{{char}}`,
historyTemplate: `GPT4 Correct {{user}}: {{message}}<|end_of_turn|>`,
char: "Assistant",
user: "User"
},



"starlingLMCode": {
template: `{{history}}{{char}}`,
historyTemplate: `Code {{user}}: {{message}}<|end_of_turn|>`,
char: "Assistant",
user: "User"
},



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



"synthia": {
template: `SYSTEM: {{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"tess": {
template: `SYSTEM: {{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"vicuna": {
template: `{{prompt}}
{{history}}
{{char}}:`,
historyTemplate: `
{{name}}: {{message}}`,
char: "ASSISTANT",
user: "USER"
},



"yi34b": {
template: `{{history}} {{char}}`,
historyTemplate: `{{name}}: {{message}}`,
char: "Assistant",
user: "Human"
},



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
