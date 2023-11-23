# train-text-from-scratch

Basic usage instructions:

        在解释train from scratch(有说简称为TFS)，即从头训练前，先说一下剪枝中的one-shot剪枝（一次剪枝）常见流程：
        训练一个大模型 -> 在大模型中剪枝 -> 微调/从头训练

        对于剪枝后的模型如何恢复精度目前有好几种方案：

        从头训练(Trrain From Scratch)：指只保留剪枝后的模型的结构，而不使用其剪枝后的权重。并随机初始化权重，再进行训练（通常使用和训练大模型时相同的学习率计划）。
        微调(Finetune)：剪枝后的模型使用小学习率继续训练。

        ggml-vocab-aquila 悟道天鹰
        ggml-vocab-falcon 不如llama2-7b
        ggml-vocab-gpt-neox EleutherAI 推出 200亿参数的类 GPT 模型：不像 GPT-3，它免费开放 GPT-NeoX-20B
        ggml-vocab-mpt MPT-7B基于1万亿tokens的文本和代码数据训练得到。是一个decoder-style类型的transformer。

```bash
# get training data
wget https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

# 构建vocab

```
spm_train --model_type=unigram --vocab_size=718 --num_threads=8 --input=slqm.txt --model_prefix=models/vocab-slqm
spm_train --model_type=unigram --vocab_size=10000 --num_threads=8 --input=hlm.txt --model_prefix=models/vocab-hlm
# --input指定需要训练的文本文件，--model_prefix指定训练好的模型名，本例中生成models/vocab-slqm.model和/models/vocab-slqm.vocab两个文件，vocab是词典信息。
# cmd 调用
echo "云对雨，雪对风，晚照对晴空。" | spm_encode --model=models/vocab-slqm.model
echo "闲言少叙，却说宝玉因近日林黛玉回去，剩得自己孤凄，也不和人顽耍，每到晚间，便索然睡了。" | spm_encode --model=models/vocab-hlm.model

```

# vocab转换为gglm格式  先将model转换为gguf,从ggml中提取vocab ???

```
python convert.py --vocab-only ../models/baichuan-inc/Baichuan2-7B-Chat/ --outfile models/ggml-vocab-bc7.gguf
python convert.py --vocab-only ../models/baichuan-inc/Baichuan2-13B-Chat/ --outfile models/ggml-vocab-bc13.gguf


```

# train
dataset="shakespeare"           \
vocab_name="ggml-vocab-llama"   \
vocab_dir="models"              \
model_dir="../models/ggmls/"    \
data_dir="../models"            \
sh examples/train-text-from-scratch/train-scratch.sh


#main: tokenize training data
./train-text-from-scratch \
        --vocab-model ../models/ggml-vocab-llama.gguf \
        --ctx 64 --embd 256 --head 8 --layer 16 \
        --checkpoint-in  chk-shakespeare-256x16-LATEST.gguf \
        --checkpoint-out chk-shakespeare-256x16-ITERATION.gguf \
        --model-out ggml-shakespeare-256x16-f32-ITERATION.gguf \
        --train-data "shakespeare.txt" \
        -t 6 -b 16 --seed 1 --adam-iter 256 \
        --no-checkpointing
./main -m ggml-shakespeare-256x16-f32.gguf

# predict
./main -m ggml-shakespeare-256x16-f32-LATEST.gguf



# 声律启蒙-baichuan
dataset="slqm"           \
vocab_name="ggml-vocab-baichuan"   \
vocab_dir="models"              \
model_dir="../models/ggmls/"    \
data_dir="../models"            \
sh examples/train-text-from-scratch/train-scratch.sh


## 训练日志

        train_opt_callback: iter=  1163 sample=1073/1600 sched=0.100000 loss=0.153680 dt=00:00:06 eta=00:00:27 |->
        train_opt_callback: iter=  1164 sample=1089/1600 sched=0.100000 loss=0.153693 dt=00:00:06 eta=00:00:20 |->
        train_opt_callback: iter=  1165 sample=1105/1600 sched=0.100000 loss=0.157189 dt=00:00:06 eta=00:00:13 |->
        train_opt_callback: iter=  1166 sample=1121/1600 sched=0.100000 loss=0.153400 dt=00:00:06 eta=00:00:06 |->
        train_opt_callback: iter=  1167 sample=1137/1600 sched=0.100000 loss=0.156294 dt=00:00:07 eta=0.0ms |->
        main: total training time: 00:31:50
        save_checkpoint_file: saving to chk-slqm-256x16-1167.gguf
        save_checkpoint_file: saving to chk-slqm-256x16-LATEST.gguf
        save_llama_model_file: saving to ggml-slqm-256x16-f32-1167.gguf
        save_llama_model_file: saving to ggml-slqm-256x16-f32-LATEST.gguf

        train_opt_callback: iter=  1929 sample=561/1600 sched=0.100000 loss=0.136846 dt=00:00:06 eta=00:00:40 |->
        train_opt_callback: iter=  1930 sample=577/1600 sched=0.100000 loss=0.137856 dt=00:00:06 eta=00:00:33 |->
        train_opt_callback: iter=  1931 sample=593/1600 sched=0.100000 loss=0.138562 dt=00:00:06 eta=00:00:26 |->
        train_opt_callback: iter=  1932 sample=609/1600 sched=0.100000 loss=0.139201 dt=00:00:06 eta=00:00:19 |->
        train_opt_callback: iter=  1933 sample=625/1600 sched=0.100000 loss=0.135440 dt=00:00:06 eta=00:00:13 |->
        train_opt_callback: iter=  1934 sample=641/1600 sched=0.100000 loss=0.141512 dt=00:00:06 eta=00:00:06 |->
        train_opt_callback: iter=  1935 sample=657/1600 sched=0.100000 loss=0.134973 dt=00:00:06 eta=0.0ms |->
        main: total training time: 01:03:53
        save_checkpoint_file: saving to chk-slqm-256x16-1935.gguf
        save_checkpoint_file: saving to chk-slqm-256x16-LATEST.gguf
        save_llama_model_file: saving to ggml-slqm-256x16-f32-1935.gguf
        save_llama_model_file: saving to ggml-slqm-256x16-f32-LATEST.gguf



# 红楼梦-baichuan 貌似macos内存不足
dataset="hlm"           \
vocab_name="ggml-vocab-baichuan"   \
vocab_dir="models"              \
model_dir="../models/ggmls/"    \
data_dir="../models"            \
sh examples/train-text-from-scratch/train-scratch.sh


## 训练日志

        train_opt_callback: iter=  4534 sample=72561/759953 sched=0.100000 loss=4.727833 dt=00:00:06 eta=00:00:13 |---->
        train_opt_callback: iter=  4535 sample=72577/759953 sched=0.100000 loss=4.911985 dt=00:00:06 eta=00:00:06 |-->
        save_checkpoint_file: saving to chk-hlm-256x16-4536.gguf
        save_checkpoint_file: saving to chk-hlm-256x16-LATEST.gguf
        save_llama_model_file: saving to ggml-hlm-256x16-f32-4536.gguf
        save_llama_model_file: saving to ggml-hlm-256x16-f32-LATEST.gguf
        train_opt_callback: iter=  4536 sample=72593/759953 sched=0.100000 loss=4.705160 dt=00:00:06 eta=0.0ms |---->
        main: total training time: 05:42:18



```

Output files will be saved every N iterations (config with `--save-every N`).
The pattern "ITERATION" in the output filenames will be replaced with the iteration number and "LATEST" for the latest output.

To train GGUF models just pass them to `--checkpoint-in FN`.
