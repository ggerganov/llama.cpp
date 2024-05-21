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
        /** @type {Object<string, SimpleChat>} */
        this.simpleChats = {};

        // the ui elements
        this.elInSystem = /** @type{HTMLInputElement} */(document.getElementById("system-in"));
        this.elDivChat = /** @type{HTMLDivElement} */(document.getElementById("chat-div"));
        this.elBtnUser = /** @type{HTMLButtonElement} */(document.getElementById("user-btn"));
        this.elInUser = /** @type{HTMLInputElement} */(document.getElementById("user-in"));
        this.elSelectApiEP = /** @type{HTMLSelectElement} */(document.getElementById("api-ep"));
        this.elDivSessions = /** @type{HTMLDivElement} */(document.getElementById("sessions-div"));

        this.validate_element(this.elInSystem, "system-in");
        this.validate_element(this.elDivChat, "chat-div");
        this.validate_element(this.elInUser, "user-in");
        this.validate_element(this.elSelectApiEP, "api-ep");
    }

    /**
     * Check if the element got
     * @param {HTMLElement | null} el
     * @param {string} msgTag
     */
    validate_element(el, msgTag) {
        if (el == null) {
            throw Error(`ERRR:MCUI:${msgTag} element missing in html...`);
        } else {
            console.debug(`INFO:MCUI:${msgTag}:Id:${el.id}:Name:${el["name"]}`);
        }
    }

    /**
     * Setup the needed callbacks wrt UI
     * @param {string} chatId
     */
    setup_ui(chatId) {
        this.elBtnUser.addEventListener("click", (ev)=>{
            if (this.elInUser.disabled) {
                return;
            }
            this.handle_user_submit(chatId, this.elSelectApiEP.value);
        });

        this.elInUser.addEventListener("keyup", (ev)=> {
            // allow user to insert enter into their message using shift+enter.
            // while just pressing enter key will lead to submitting.
            if ((ev.key === "Enter") && (!ev.shiftKey)) {
                this.elBtnUser.click();
                ev.preventDefault();
            }
        });
    }

    /**
     * Start a new chat session
     * @param {string} chatId
     */
    new_chat(chatId) {
        this.simpleChats[chatId] = new SimpleChat();
    }

    /**
     * Handle user query submit request, wrt specified chat session.
     * @param {string} chatId
     * @param {string} apiEP
     */
    async handle_user_submit(chatId, apiEP) {

        let chat = this.simpleChats[chatId];

        chat.add_system_anytime(this.elInSystem.value, chatId);

        let content = this.elInUser.value;
        if (!chat.add(Roles.User, content)) {
            console.debug(`WARN:MCUI:${chatId}:HandleUserSubmit:Ignoring empty user input...`);
            return;
        }
        chat.show(this.elDivChat);

        let theBody;
        let theUrl = gChatURL[apiEP]
        if (apiEP == ApiEP.Chat) {
            theBody = chat.request_messages_jsonstr();
        } else {
            theBody = chat.request_prompt_jsonstr();
        }

        this.elInUser.value = "working...";
        this.elInUser.disabled = true;
        console.debug(`DBUG:MCUI:${chatId}:HandleUserSubmit:${theUrl}:ReqBody:${theBody}`);
        let resp = await fetch(theUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: theBody,
        });

        this.elInUser.value = "";
        this.elInUser.disabled = false;
        let respBody = await resp.json();
        console.debug(`DBUG:MCUI:${chatId}:HandleUserSubmit:RespBody:${JSON.stringify(respBody)}`);
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
        chat.show(this.elDivChat);
        // Purposefully clear at end rather than begin of this function
        // so that one can switch from chat to completion mode and sequece
        // in a completion mode with multiple user-assistant chat data
        // from before to be sent/occur once.
        if ((apiEP == ApiEP.Completion) && (gbCompletionFreshChatAlways)) {
            chat.xchat.length = 0;
        }
        this.elInUser.focus();
    }

    /**
     * Show buttons for available chat sessions, in the passed elDiv.
     * If elDiv is undefined/null, then use this.elDivSessions.
     * @param {HTMLDivElement | undefined} elDiv
     */
    show_sessions(elDiv=undefined) {
        if (!elDiv) {
            elDiv = this.elDivSessions;
        }
        elDiv.replaceChildren();
        let chatIds = Object.keys(this.simpleChats);
        for(let cid of chatIds) {
            let btn = document.createElement("button");
            btn.id = cid;
            btn.name = cid;
            btn.innerText = cid;
            btn.addEventListener("click", (ev)=>{
                let target = /** @type{HTMLButtonElement} */(ev.target);
                console.debug(`DBUG:MCUI:SessionClick:${target.id}`);
                if (this.elInUser.disabled) {
                    console.error(`ERRR:MCUI:SessionClick:${target.id}:Current session awaiting response, skipping switch...`);
                    return;
                }
                this.handle_session_switch(target.id);
            });
            elDiv.appendChild(btn);
        }
    }

    /**
     * @param {string} chatId
     */
    async handle_session_switch(chatId) {
        let chat = this.simpleChats[chatId];
        if (chat == undefined) {
            console.error(`ERRR:MCUI:Session:${chatId} missing...`);
            return;
        }
        this.elInUser.value = "";
        chat.show(this.elDivChat);
        this.elInUser.focus();
    }

}


let gMuitChat;
const gChatId = "Default";

function startme() {
    console.log("INFO:StartMe:Starting...");
    gMuitChat = new MultiChatUI();
    gMuitChat.new_chat(gChatId);
    gMuitChat.setup_ui(gChatId);
    gMuitChat.show_sessions();
}

document.addEventListener("DOMContentLoaded", startme);
