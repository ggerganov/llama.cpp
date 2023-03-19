#!/usr/bin/env bash

PORT=${PORT:-8080}
PROMPT="${PROMPT:-"Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User:Hello, Bob.
Bob:Hello. How may I help you today?
User:Please tell me the largest city in Europe.
Bob:Sure. The largest city in Europe is Moscow, the capital of Russia.
User:"}"
RPROMPT="${RPROMPT:-"User:"}"
N_PREDICT="${N_PREDICT:-"4096"}"
REPEAT_PENALTY="${REPEAT_PENALTY:-"1.0"}"
N_THREADS="${N_THREADS:-"4"}"

# Open connection to the chat server
exec 3<>/dev/tcp/127.0.0.1/${PORT}

# Pass the arguments. The protocol is really simple:
# 1. Pass the number of arguments followed by a linefeed
# 2. Pass the arguments, with each being followed by "0"
(
echo -en "12\n"
echo -en "-t\x00"
echo -en "$N_THREADS\x00"
echo -en "-n\x00"
echo -en "$N_PREDICT\x00"
echo -en "--repeat_penalty\x00"
echo -en "$REPEAT_PENALTY\x00"
echo -en "--color\x00"
echo -en "-i\x00"
echo -en "-r\x00"
echo -en "$RPROMPT\x00"
echo -en "-p\x00"
echo -en "$PROMPT\x00"
) >&3

trap exit TERM

# When we have passed the arguments, start printing socket data to the screen.
# This is done in a background job because we also want to send data when
# running in interactive mode.
cat <&3 && echo "(disconnected, press \"enter\" twice to exit)" &
cat >&3
wait
