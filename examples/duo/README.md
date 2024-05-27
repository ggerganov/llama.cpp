## duo

This is a demo of an approach of distributed evaluation/speculation using rpc.

Idea is coming from discussion here: https://github.com/ggerganov/llama.cpp/discussions/6853#discussioncomment-9473494.
When we run a large model and distribute the evaluation across multiple instances, they still evaluate model sequentially in case of individial query/no pipelining. 
In case of two identical devices and equal model split we would leave half of compute on the table.

We can utilize this compute to speculate and then evaluate larger sequence of tokens. 

This demo is fairly limited, more like a proof of concept:
1. Expects exactly two instances running main model
2. Only one of these instances speculating when main model is idle, so we still waste 25% of compute. Once we get a callback that a split is done, the instance running that split becomes idle and we start running speculation model there until main model becomes active again.
3. Speculation is linear
4. Sampling is greedy

Improvement of the above points is probably easier to do as separate changes, to make reviewing and testing easier.

### Setup

Devices:
* Apple M1 16GB 
* Apple M2 24GB
* Connected with thunderbolt-4 cable and using IP over thunderbolt. 

Models:
* Meta-Llama-3-8B-Instruct-fp16 as main
* Meta-Llama-3-8B-Instruct-v2.Q2_K as speculation

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
llama_print_timings:        load time =   42068.04 ms
...
llama_print_timings:       total time =   42792.74 ms /   302 tokens

```

Seems like eval time is messed up a little 

Compare that with running main with same 2 rpc servers:
```
./bin/main -m ../../llms/gguf/Meta-Llama-3-8B-Instruct-fp16.gguf  --rpc "localhost:10001,169.254.77.16:10001" -p "Please illustrate the difference between concurrency and parallelism in python." -n 256 -ngl 99
...
llama_print_timings:        load time =   42305.61 ms
...
llama_print_timings:       total time =   58555.49 ms /   268 tokens
```

Extra: 

GPU util for both devices

<img width="1350" alt="Screenshot 2024-05-27 at 12 42 34 PM" src="https://github.com/okuvshynov/llama.cpp/assets/661042/2275506d-ef3c-4cc0-9853-cb00354cc06d">

In duo case: we utilize GPU at ~100% for instance running both speculation and main model, and ~50% for the one running main model only
In main model only case: we utilize both at ~50%. The imbalance is likely because hardware is slightly different - M2 vs M1.
