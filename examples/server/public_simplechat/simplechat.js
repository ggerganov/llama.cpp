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
    }

    /**
     * Add an entry into xchat
     * @param {string} role
     * @param {string|undefined|null} content
     */
    add(role, content) {
        if ((content == undefined) || (content == null) || (content == "")) {
            return;
        }
        this.xchat.push( {role: role, content: content} );
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
        for(const x of this.xchat) {
            let entry = document.createElement("p");
            entry.className = `role-${x.role}`;
            entry.innerText = `${x.role}: ${x.content}`;
            div.appendChild(entry);
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

}

/**
 * Handle submit request by user
 * @param {HTMLInputElement} inputUser
 * @param {HTMLDivElement} divChat
 * @param {string} apiEP
 */
async function handle_submit(inputUser, divChat, apiEP) {

    let content = inputUser?.value;
    gChat.add(Roles.User, content);
    gChat.show(divChat);

    let theBody;
    let theUrl = gChatURL[apiEP]
    if (apiEP == ApiEP.Chat) {
        theBody = gChat.request_messages_jsonstr();
    } else {
        theBody = gChat.request_prompt_jsonstr();
    }

    inputUser.scrollIntoView(true);
    inputUser.value = "working...";
    inputUser.disabled = true;
    console.debug(`DBUG:HandleSubmit:${theUrl}:ReqBody:${theBody}`);
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
    console.debug("DBUG:HandleSubmit:RespBody:", respBody);
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
    gChat.add(Roles.Assistant, assistantMsg);
    gChat.show(divChat);
    inputUser.scrollIntoView(true);

}


let gChat = new SimpleChat();
let gBaseURL = "http://127.0.0.1:8080";
let gChatURL = {
    'chat': `${gBaseURL}/chat/completions`,
    'completion': `${gBaseURL}/completions`,
}


function startme() {

    let divChat = /** @type{HTMLDivElement} */(document.getElementById("chat"));
    let btnSubmit = document.getElementById("submit");
    let inputUser = /** @type{HTMLInputElement} */(document.getElementById("user"));
    let selectApiEP = /** @type{HTMLInputElement} */(document.getElementById("api-ep"));

    if (divChat == null) {
        throw Error("ERRR:StartMe:Chat element missing");
    }

    btnSubmit?.addEventListener("click", (ev)=>{
        if (inputUser.disabled) {
            return;
        }
        handle_submit(inputUser, divChat, selectApiEP.value);
    });

    inputUser?.addEventListener("keyup", (ev)=> {
        if (ev.key === "Enter") {
            btnSubmit?.click();
            ev.preventDefault();
        }
    });

}


document.addEventListener("DOMContentLoaded", startme);
