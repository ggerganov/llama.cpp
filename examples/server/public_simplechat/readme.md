
# SimpleChat

by Humans for All.


## overview

This simple web frontend, allows triggering/testing the server's /completions or /chat/completions endpoints
in a simple way with minimal code from a common code base. Inturn additionally it tries to allow single or
multiple independent back and forth chatting to an extent, with the ai llm model at a basic level, with their
own system prompts.

The UI follows a responsive web design so that the layout can adapt to available display space in a usable
enough manner, in general.

Allows developer/end-user to control some of the behaviour by updating gMe members from browser's devel-tool
console.

NOTE: Given that the idea is for basic minimal testing, it doesnt bother with any model context length and
culling of old messages from the chat by default. However by enabling the sliding window chat logic, a crude
form of old messages culling can be achieved.

NOTE: It doesnt set any parameters other than temperature and max_tokens for now. However if someone wants
they can update the js file or equivalent member in gMe as needed.


## usage

One could run this web frontend directly using server itself or if anyone is thinking of adding a built in web
frontend to configure the server over http(s) or so, then run this web frontend using something like python's
http module.

### running using examples/server

bin/server -m path/model.gguf --path ../examples/server/public_simplechat [--port PORT]

### running using python3's server module

first run examples/server
* bin/server -m path/model.gguf

next run this web front end in examples/server/public_simplechat
* cd ../examples/server/public_simplechat
* python3 -m http.server PORT

### using the front end

Open this simple web front end from your local browser

* http://127.0.0.1:PORT/index.html

Once inside

* Select between chat and completion mode. By default it is set to chat mode.

* In completion mode
  * logic by default doesnt insert any role specific "ROLE: " prefix wrt each role's message.
    If the model requires any prefix wrt user role messages, then the end user has to
    explicitly add the needed prefix, when they enter their chat message.
    Similarly if the model requires any prefix to trigger assistant/ai-model response,
    then the end user needs to enter the same.
    This keeps the logic simple, while still giving flexibility to the end user to
    manage any templating/tagging requirement wrt their messages to the model.
  * the logic doesnt insert newline at the begining and end wrt the prompt message generated.
    However if the chat being sent to /completions end point has more than one role's message,
    then insert newline when moving from one role's message to the next role's message, so
    that it can be clearly identified/distinguished.
  * given that /completions endpoint normally doesnt add additional chat-templating of its
    own, the above ensures that end user can create a custom single/multi message combo with
    any tags/special-tokens related chat templating to test out model handshake. Or enduser
    can use it just for normal completion related/based query.

* If you want to provide a system prompt, then ideally enter it first, before entering any user query.
  Normally Completion mode doesnt need system prompt, while Chat mode can generate better/interesting
  responses with a suitable system prompt.
  * if chat.add_system_begin is used
    * you cant change the system prompt, after it is has been submitted once along with user query.
    * you cant set a system prompt, after you have submitted any user query
  * if chat.add_system_anytime is used
    * one can change the system prompt any time during chat, by changing the contents of system prompt.
    * inturn the updated/changed system prompt will be inserted into the chat session.
    * this allows for the subsequent user chatting to be driven by the new system prompt set above.

* Enter your query and either press enter or click on the submit button.
  If you want to insert enter (\n) as part of your chat/query to ai model, use shift+enter.

* Wait for the logic to communicate with the server and get the response.
  * the user is not allowed to enter any fresh query during this time.
  * the user input box will be disabled and a working message will be shown in it.

* just refresh the page, to reset wrt the chat history and or system prompt and start afresh.

* Using NewChat one can start independent chat sessions.
  * two independent chat sessions are setup by default.


## Devel note

### Reason behind this

The idea is to be easy enough to use for basic purposes, while also being simple and easily discernable
by developers who may not be from web frontend background (so inturn may not be familiar with template /
end-use-specific-language-extensions driven flows) so that they can use it to explore/experiment things.

And given that the idea is also to help explore/experiment for developers, some flexibility is provided
to change behaviour easily using the devel-tools/console, for now. And skeletal logic has been implemented
to explore some of the end points and ideas/implications around them.


### General

Me/gMe consolidates the settings which control the behaviour into one object.
One can see the current settings, as well as change/update them using browsers devel-tool/console.

  bCompletionFreshChatAlways - whether Completion mode collates complete/sliding-window history when
  communicating with the server or only sends the latest user query/message.

  bCompletionInsertStandardRolePrefix - whether Completion mode inserts role related prefix wrt the
  messages that get inserted into prompt field wrt /Completion endpoint.

  chatRequestOptions - maintains the list of options/fields to send along with chat request,
  irrespective of whether /chat/completions or /completions endpoint.

    If you want to add additional options/fields to send to the server/ai-model, and or
    modify the existing options value or remove them, for now you can update this global var
    using browser's development-tools/console.

  iRecentUserMsgCnt - a simple minded SlidingWindow to limit context window load at Ai Model end.
  This is disabled by default. However if enabled, then in addition to latest system message, only
  the last/latest iRecentUserMsgCnt user messages after the latest system prompt and its responses
  from the ai model will be sent to the ai-model, when querying for a new response. IE if enabled,
  only user messages after the latest system message/prompt will be considered.

    This specified sliding window user message count also includes the latest user query.
    <0 : Send entire chat history to server
     0 : Send only the system message if any to the server
    >0 : Send the latest chat history from the latest system prompt, limited to specified cnt.


By using gMe's iRecentUserMsgCnt and chatRequestOptions.max_tokens one can try to control the
implications of loading of the ai-model's context window by chat history, wrt chat response to
some extent in a simple crude way.


Sometimes the browser may be stuborn with caching of the file, so your updates to html/css/js
may not be visible. Also remember that just refreshing/reloading page in browser or for that
matter clearing site data, dont directly override site caching in all cases. Worst case you may
have to change port. Or in dev tools of browser, you may be able to disable caching fully.


Concept of multiple chat sessions with different servers, as well as saving and restoring of
those across browser usage sessions, can be woven around the SimpleChat/MultiChatUI class and
its instances relatively easily, however given the current goal of keeping this simple, it has
not been added, for now.


By switching between chat.add_system_begin/anytime, one can control whether one can change
the system prompt, anytime during the conversation or only at the beginning.


read_json_early, is to experiment with reading json response data early on, if available,
so that user can be shown generated data, as and when it is being generated, rather than
at the end when full data is available.

  the server flow doesnt seem to be sending back data early, atleast for request (inc options)
  that is currently sent.

  if able to read json data early on in future, as and when ai model is generating data, then
  this helper needs to indirectly update the chat div with the recieved data, without waiting
  for the overall data to be available.


### Default setup

By default things are setup to try and make the user experience a bit better, if possible.
However a developer when testing the server of ai-model may want to change these value.

Using iRecentUserMsgCnt reduce chat history context sent to the server/ai-model to be
just the system-prompt, prev-user-request-and-ai-response and cur-user-request, instead of
full chat history. This way if there is any response with garbage/repeatation, it doesnt
mess with things beyond the next question/request/query, in some ways.

Set max_tokens to 1024, so that a relatively large previous reponse doesnt eat up the space
available wrt next query-response. However dont forget that the server when started should
also be started with a model context size of 1k or more, to be on safe side.

  The /completions endpoint of examples/server doesnt take max_tokens, instead it takes the
  internal n_predict, for now add the same here on the client side, maybe later add max_tokens
  to /completions endpoint handling code on server side.

Frequency and presence penalty fields are set to 1.2 in the set of fields sent to server
along with the user query. So that the model is partly set to try avoid repeating text in
its response.

A end-user can change these behaviour by editing gMe from browser's devel-tool/console.


## At the end

Also a thank you to all open source and open model developers, who strive for the common good.
