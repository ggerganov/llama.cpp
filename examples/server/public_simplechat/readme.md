
# SimpleChat

by Humans for All.

## overview

This simple web frontend, allows triggering/testing the server's /completions or /chat/completions endpoints
in a simple way with minimal code from a common code base. And also allows trying to maintain a basic back
and forth chatting to an extent.

User can set a system prompt, as well as try and chat with the model over a series of back and forth chat
messages.

NOTE: Given that the idea is for basic minimal testing, it doesnt bother with any model context length and
culling of old messages from the chat.

NOTE: It doesnt set any parameters other than temperature for now. However if someone wants they can update
the js file as needed.

## usage

first run examples/server
* bin/server -m path/model.gguf

next run this web front end in examples/server/public_simplechat
* ./simplechat.sh
* this uses python3's http.server to host this web front end

Open this simple web front end from your local browser as noted in the message printed when simplechat.sh is run
* by default it is http://127.0.0.1:9000/simplechat.html

Once inside
* Select between chat and completion mode. By default it is set to chat mode.
* If you want to provide a system prompt, then enter it first, before entering any user query.
  * you cant change the system prompt, after it is has been submitted once along with user query.
  * you cant set a system prompt, after you have submitted any user query
* Enter your query and either press enter or click on the submit button
* Wait for the logic to communicate with the server and get the response.
  * the user is not allowed to enter any fresh query during this time.
  * the user input box will be disabled and a working message will be shown in it.
* just refresh the page, to reset wrt the chat history and or system prompt and start afresh.


## Devel note

Sometimes the browser may be stuborn with caching of the file, so your updates to html/css/js
may not be visible. Also remember that just refreshing/reloading page in browser or for that
matter clearing site data, dont directly override site caching in all cases. Worst case you may
have to change port.

