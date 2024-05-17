// @ts-check

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
     * @param {string} content
     */
    add(role, content) {
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
    let divChat = document.getElementById("chat");
    let btnSubmit = document.getElementById()
}
