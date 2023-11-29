# TODO List
  微调流程已通；
  选择合适的行业场景；
  测试微调文本格式与效果；
  百川模型转换为llama;
  好的中文llama base model;

# finetune

Basic usage instructions:

```bash
# get training data
wget https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

# finetune LORA adapter
# ./bin/finetune \
#         --model-base open-llama-3b-v2-q8_0.gguf \
#         --checkpoint-in  chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf \
#         --checkpoint-out chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf \
#         --lora-out lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin \
#         --train-data "shakespeare.txt" \
#         --save-every 10 \
#         --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
#         --use-checkpointing

# # predict
# ./bin/main -m open-llama-3b-v2-q8_0.gguf --lora lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin

# mac os mistral
dataset=slqm \
data_dir="../models/"  \
model_dir="../models/ggmls/" \
model_name="openbuddy-mistral-7b-v13.1-q2_k" \
sh examples/finetune/finetune.sh >${model_name}.log

#云电脑   mistral
dataset=slqm \
data_dir="../"  \
model_dir="../"  \
model_name="openbuddy-mistral-7b-v13.1-q2_k" \
sh examples/finetune/finetune.sh >${model_name}.log

# mac os llama2
dataset=slqm \
data_dir="../models/"  \
model_dir="../models/ggmls/" \
model_name="chinese-llama-2-7b-16k.Q2_K" \
sh examples/finetune/finetune.sh >${model_name}.log

#云电脑   llama2
dataset="slqm" \
data_dir="../"  \
model_dir="../"  \
model_name="chinese-llama-2-7b-16k.Q2_K" \
sh examples/finetune/finetune.sh >${model_name}.log


#云电脑   finetune-bc
dataset="slqm" \
data_dir="../"  \
model_dir="../"  \
model_name="bc2-7b-chat-q2_k" \
cmd="finetune-bc" \
sh examples/finetune/finetune.sh >${model_name}.log

dataset="slqm" \
data_dir="../"  \
model_dir="../"  \
model_name="bc2-13b-chat-q2_k" \
adamiter=90 \
cmd="finetune-bc" \
inter=true \
sh examples/finetune/finetune.sh >bc2-13b-chat-q2_k.log

# macos
dataset="slqm" \
data_dir="../models/"  \
model_dir="../models/ggmls/"  \
model_name="bc2-13b-chat-q2_k" \
cmd="finetune-bc" \
inter=1 \
sh examples/finetune/finetune.sh >bc2-13b-chat-q2_k.log


# ./main -m ../models/ggmls/openbuddy-llama2-70b-v10.1-bf16-q3_k_s.gguf -n 128     -p '"人间清暑殿，天上广寒宫。"的下一句'


1.微调 CPU核素越多越快 只支持llama架构
win cmd 输出编码
    $PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
修改Windows10 命令终端cmd的编码为UTF-8
    chcp 65001

```

Finetune output files will be saved every N iterations (config with `--save-every N`).
The pattern 'ITERATION' in the output filenames will be replaced with the iteration number and with 'LATEST' for the latest output.
So in above example after 10 iterations these files will be written:
- chk-lora-open-llama-3b-v2-q8_0-shakespeare-10.gguf
- chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf
- lora-open-llama-3b-v2-q8_0-shakespeare-10.bin
- lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin

After 10 more iterations:
- chk-lora-open-llama-3b-v2-q8_0-shakespeare-20.gguf
- chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf
- lora-open-llama-3b-v2-q8_0-shakespeare-20.bin
- lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin

Checkpoint files (`--checkpoint-in FN`, `--checkpoint-out FN`) store the training process. When the input checkpoint file does not exist, it will begin finetuning a new randomly initialized adapter.

llama.cpp compatible LORA adapters will be saved with filename specified by `--lora-out FN`.
These LORA adapters can then be used by `main` together with the base model, like in the 'predict' example command above.

In `main` you can also load multiple LORA adapters, which will then be mixed together.

For example if you have two LORA adapters `lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin` and `lora-open-llama-3b-v2-q8_0-bible-LATEST.bin`, you can mix them together like this:

```bash
./bin/main -m open-llama-3b-v2-q8_0.gguf \
  --lora lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin \
  --lora lora-open-llama-3b-v2-q8_0-bible-LATEST.bin
```

You can change how strong each LORA adapter is applied to the base model by using `--lora-scaled FN SCALE` instead of `--lora FN`.

For example to apply 40% of the 'shakespeare' LORA adapter, 80% of the 'bible' LORA adapter and 100% of yet another one:

```bash
./bin/main -m open-llama-3b-v2-q8_0.gguf \
  --lora-scaled lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin 0.4 \
  --lora-scaled lora-open-llama-3b-v2-q8_0-bible-LATEST.bin 0.8 \
  --lora lora-open-llama-3b-v2-q8_0-yet-another-one-LATEST.bin
```

The scale numbers don't need to add up to one, and you can also use numbers greater than 1 to further increase the influence of an adapter. But making the values to big will sometimes result in worse output. Play around to find good values.

Gradient checkpointing reduces the memory requirements by ~50% but increases the runtime.
If you have enough RAM, you can make finetuning a bit faster by disabling checkpointing with `--no-checkpointing`.

The default LORA rank can be specified with `--lora-r N`.
The LORA rank can be configured for each model tensor type separately with these command line options:

```bash
  --lora-r N                 LORA r: default rank. Also specifies resulting scaling together with lora-alpha. (default 4)
  --rank-att-norm N          LORA rank for attention norm tensor (default 1)
  --rank-ffn-norm N          LORA rank for feed-forward norm tensor (default 1)
  --rank-out-norm N          LORA rank for output norm tensor (default 1)
  --rank-tok-embd N          LORA rank for token embeddings tensor (default 4)
  --rank-out N               LORA rank for output tensor (default 4)
  --rank-wq N                LORA rank for wq tensor (default 4)
  --rank-wk N                LORA rank for wk tensor (default 4)
  --rank-wv N                LORA rank for wv tensor (default 4)
  --rank-wo N                LORA rank for wo tensor (default 4)
  --rank-w1 N                LORA rank for w1 tensor (default 4)
  --rank-w2 N                LORA rank for w2 tensor (default 4)
  --rank-w3 N                LORA rank for w3 tensor (default 4)
```

The LORA rank of 'norm' tensors should always be 1.

To see all available options use `finetune --help`.
