// @ts-check

class Roles {
    static System = "system";
    static User = "user";
    static Assistant = "assistant";
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
            entry.innerText = `${x.role}: ${x.content}`;
            div.appendChild(entry);
        }
    }

    request_json() {
        let req = {
            messages: this.xchat,
            temperature: 0.7
        }
        return JSON.stringify(req);
    }

}


let gChat = new SimpleChat();
let gBaseURL = "http://127.0.0.1:8080";
let gChatURL = `${gBaseURL}/chat/completions`;


function startme() {

    let divChat = /** @type{HTMLDivElement} */(document.getElementById("chat"));
    let btnSubmit = document.getElementById("submit");
    let inputUser = /** @type{HTMLInputElement} */(document.getElementById("user"));

    if (divChat == null) {
        throw Error("ERRR:StartMe:Chat element missing");
    }

    btnSubmit?.addEventListener("click", (ev)=>{
        let content = inputUser?.value;
        console.debug("DBUG:BtnSubmit:Click:", content)
        gChat.add(Roles.User, content);
        gChat.show(divChat);
    });

}

document.addEventListener("DOMContentLoaded", startme);
