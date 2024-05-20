// @ts-check
// A simple completions and chat/completions test related web front end logic
// by Humans for All

class Roles {
    static System = "system";
    static User = "user";
    static Assistant = "assistant";
}

class ApiEP {
    static Chat = "chat";
    static Completion = "completion";
}

class SimpleChat {

    constructor() {
        /**
         * Maintain in a form suitable for common LLM web service chat/completions' messages entry
         * @type {{role: string, content: string}[]}
         */
        this.xchat = [];
        this.iLastSys = -1;
    }

    /**
     * Add an entry into xchat
     * @param {string} role
     * @param {string|undefined|null} content
     */
    add(role, content) {
        if ((content == undefined) || (content == null) || (content == "")) {
            return false;
        }
        this.xchat.push( {role: role, content: content} );
        if (role == Roles.System) {
            this.iLastSys = this.xchat.length - 1;
        }
        return true;
    }

    /**
     * Show the contents in the specified div
     * @param {HTMLDivElement} div
     * @param {boolean} bClear
     */
    show(div, bClear=true) {
        if (bClear) {
            div.replaceChildren();
        }
        let last = undefined;
        for(const x of this.xchat) {
            let entry = document.createElement("p");
            entry.className = `role-${x.role}`;
            entry.innerText = `${x.role}: ${x.content}`;
            div.appendChild(entry);
            last = entry;
        }
        if (last !== undefined) {
            last.scrollIntoView(false);
        }
    }

    /**
     * Add needed fields wrt json object to be sent wrt LLM web services completions endpoint
     * Convert the json into string.
     * @param {Object} obj
     */
    request_jsonstr(obj) {
        obj["temperature"] = 0.7;
        return JSON.stringify(obj);
    }

    /**
     * Return a string form of json object suitable for chat/completions
     */
    request_messages_jsonstr() {
        let req = {
            messages: this.xchat,
        }
        return this.request_jsonstr(req);
    }

    /**
     * Return a string form of json object suitable for /completions
     */
    request_prompt_jsonstr() {
        let prompt = "";
        for(const chat of this.xchat) {
            prompt += `${chat.role}: ${chat.content}\n`;
        }
        let req = {
            prompt: prompt,
        }
        return this.request_jsonstr(req);
    }

    /**
     * Allow setting of system prompt, but only at begining.
     * @param {string} sysPrompt
     * @param {string} msgTag
     */
    add_system_begin(sysPrompt, msgTag) {
        if (this.xchat.length == 0) {
            if (sysPrompt.length > 0) {
                return this.add(Roles.System, sysPrompt);
            }
        } else {
            if (sysPrompt.length > 0) {
                if (this.xchat[0].role !== Roles.System) {
                    console.error(`ERRR:SimpleChat:${msgTag}:You need to specify system prompt before any user query, ignoring...`);
                } else {
                    if (this.xchat[0].content !== sysPrompt) {
                        console.error(`ERRR:SimpleChat:${msgTag}:You cant change system prompt, mid way through, ignoring...`);
                    }
                }
            }
        }
        return false;
    }

    /**
     * Allow setting of system prompt, at any time.
     * @param {string} sysPrompt
     * @param {string} msgTag
     */
    add_system_anytime(sysPrompt, msgTag) {
        if (sysPrompt.length <= 0) {
            return false;
        }

        if (this.iLastSys < 0) {
            return this.add(Roles.System, sysPrompt);
        }

        let lastSys = this.xchat[this.iLastSys].content;
        if (lastSys !== sysPrompt) {
            return this.add(Roles.System, sysPrompt);
        }
        return false;
    }

}


let gBaseURL = "http://127.0.0.1:8080";
let gChatURL = {
    'chat': `${gBaseURL}/chat/completions`,
    'completion': `${gBaseURL}/completions`,
}
const gbCompletionFreshChatAlways = true;


class MultiChatUI {

    constructor() {
        /** @type {number} */
        this.iChat = -1;
        /** @type {SimpleChat[]} */
        this.simpleChats = [];
    }

    /**
     * Start a new chat session
     */
    new_chat() {
        this.simpleChats.push(new SimpleChat());
        this.iChat = this.simpleChats.length - 1;
    }

    /**
     * Handle user query submit request, wrt current chat session.
     * @param {HTMLInputElement} inputSystem
     * @param {HTMLInputElement} inputUser
     * @param {HTMLDivElement} divChat
     * @param {string} apiEP
     */
    async handle_user_submit(inputSystem, inputUser, divChat, apiEP) {

        let chat = this.simpleChats[this.iChat];

        chat.add_system_anytime(inputSystem.value, "0");

        let content = inputUser.value;
        if (!chat.add(Roles.User, content)) {
            console.debug("WARN:MCUI:HandleUserSubmit:Ignoring empty user input...");
            return;
        }
        chat.show(divChat);

        let theBody;
        let theUrl = gChatURL[apiEP]
        if (apiEP == ApiEP.Chat) {
            theBody = chat.request_messages_jsonstr();
        } else {
            theBody = chat.request_prompt_jsonstr();
        }

        inputUser.value = "working...";
        inputUser.disabled = true;
        console.debug(`DBUG:MCUI:HandleUserSubmit:${theUrl}:ReqBody:${theBody}`);
        let resp = await fetch(theUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: theBody,
        });

        inputUser.value = "";
        inputUser.disabled = false;
        let respBody = await resp.json();
        console.debug("DBUG:MCUI:HandleUserSubmit:RespBody:", respBody);
        let assistantMsg;
        if (apiEP == ApiEP.Chat) {
            assistantMsg = respBody["choices"][0]["message"]["content"];
        } else {
            try {
                assistantMsg = respBody["choices"][0]["text"];
            } catch {
                assistantMsg = respBody["content"];
            }
        }
        chat.add(Roles.Assistant, assistantMsg);
        chat.show(divChat);
        // Purposefully clear at end rather than begin of this function
        // so that one can switch from chat to completion mode and sequece
        // in a completion mode with multiple user-assistant chat data
        // from before to be sent/occur once.
        if ((apiEP == ApiEP.Completion) && (gbCompletionFreshChatAlways)) {
            chat.xchat.length = 0;
        }
        inputUser.focus();
    }

}


let gMuitChat = new MultiChatUI();


function startme() {

    let inputSystem = /** @type{HTMLInputElement} */(document.getElementById("system"));
    let divChat = /** @type{HTMLDivElement} */(document.getElementById("chat"));
    let btnSubmit = document.getElementById("submit");
    let inputUser = /** @type{HTMLInputElement} */(document.getElementById("user"));
    let selectApiEP = /** @type{HTMLInputElement} */(document.getElementById("api-ep"));

    if (divChat == null) {
        throw Error("ERRR:StartMe:Chat element missing");
    }

    gMuitChat.new_chat();

    btnSubmit?.addEventListener("click", (ev)=>{
        if (inputUser.disabled) {
            return;
        }
        gMuitChat.handle_user_submit(inputSystem, inputUser, divChat, selectApiEP.value);
    });

    inputUser?.addEventListener("keyup", (ev)=> {
        // allow user to insert enter into their message using shift+enter.
        // while just pressing enter key will lead to submitting.
        if ((ev.key === "Enter") && (!ev.shiftKey)) {
            btnSubmit?.click();
            ev.preventDefault();
        }
    });

}


document.addEventListener("DOMContentLoaded", startme);
