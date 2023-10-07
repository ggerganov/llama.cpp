# llama.cpp/example/parallel

Simplified simulation for serving incoming requests in parallel

Running this using the 100 questions in examples/jeopardy/questions.txt
on an M2 MAX (38 core) with 32GB unified memory on MacOS Sonoma 14.0
takes about 235 seconds with sequential responses (-ns 1) and 45 seconds
with 64 parallel responses (-ns 64) in both cases generating 100 answers (-np 100)
using a context of 8192 (-c 8192).
