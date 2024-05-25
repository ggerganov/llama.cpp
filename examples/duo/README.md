## duo

This is a demo of an approach of distributed evaluation/speculation using rpc.

It is a fairly minimal app, and many more improvements could be made. 

### Idea

Idea is coming from discussion here: https://github.com/ggerganov/llama.cpp/discussions/6853#discussioncomment-9473494.
When we run a large model and distribute the evaluation across multiple devices, they still evaluate model sequentially.
In case of two identical devices and equal model split we would leave half of compute on the table, assuming individual use-case (e.g. personal chat).

We can utilize this compute to speculate and then evaluate larger sequence of tokens.

This demo is fairly limited:
1. Expects two instances running main model
2. One of these instances speculating
3. Speculation is linear
4. Sampling is greedy

So, in the case of two identical devices and equal model split we still are not utilizing 25% of compute.
Improvement of the above points is probably easier to do as separate changes, to make reviewing easier.

### Setup

Devices:
* Apple M1 16GB 
* Apple M2 24GB
* Connected with thunderbolt-4 cable and using TCP/IP over thunderbolt. 

Models:
* Meta-Llama-3-8B-Instruct-fp16 as main
* Meta-Llama-3-8B-Instruct-v2.Q2_K as speculation

We could use different models as well.

On M1
```
bin/rpc-server -p 10001 -m 10000
```

On M2
```
bin/rpc-server -p 10001 -m 10000
bin/rpc-server -p 20002 -m 4000
```

Also on M2:
```
./bin/duo -m ../../llms/gguf/Meta-Llama-3-8B-Instruct-fp16.gguf -md ../../llms/gguf/Meta-Llama-3-8B-Instruct-v2.Q2_K.gguf --rpc "localhost:10001,169.254.77.16:10001" -p "Please illustrate the difference between concurrency and parallelism in python." -n 256 -ngl 99 -t 1  --rpcd "localhost:20002"

...
decoded 256 tokens in 32.03 s, speed: 7.99 t/s

```

Compare that with running main with same 2 rpc servers:
```
./bin/main -m ../../llms/gguf/Meta-Llama-3-8B-Instruct-fp16.gguf  --rpc "localhost:10001,169.254.77.16:10001" -p "Please illustrate the difference between concurrency and parallelism in python." -n 256 -ngl 99 -t 1
...

```


