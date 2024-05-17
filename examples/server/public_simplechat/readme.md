
# SimpleChat

by Humans for All.

## overview

This simple web frontend, allows triggering/testing the server's /completions or /chat/completions endpoints
in a simple way with minimal code from a common code base. And also allows trying to maintain a basic back
and forth chatting to an extent.

NOTE: Given that the idea is for basic minimal testing, it doesnt bother with any model context length and
culling of old messages from the chat. Also currently I havent added input for a system prompt, but may add it.

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

