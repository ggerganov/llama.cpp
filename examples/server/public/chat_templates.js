export const chat_templates = {
        "Alpaca" : {
            "template" : "{{prompt}}\n### Instruction:\n{{history}}\n### Response:\n{{char}}:",
            "historyTemplate" : "{{name}}: {{message}}"
        },

        "Vicuna" : {
            "template" : "{{prompt}}\n\n{{history}}\n{{char}}:",
            "historyTemplate" : "{{name}}: {{message}}"
        },
        
        "ChatML" : {
            "template" : "<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}<|im_start|>{{char}}\n",
            "historyTemplate" : "<|im_start|>{{name}}\n{{message}}<|im_end|>\n"
        }
};
