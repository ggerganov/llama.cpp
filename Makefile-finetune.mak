# 在mac & win 都能进行微调；
# 能够微调不通的模型；
# 能够使用不同的数据集；

# 默认使用finetune;百川使用finetune-bc
adamcnt = 30
finetune = finetune
model_dir = ../models/ggmls
data_dir = ../models
dataset = slqm
prompt='"人间清暑殿，天上广寒宫。"的下一句'

finetunel4lora:
	echo "start model finetune ......"
	./${finetune} \
		--train-data ${data_dir}/${dataset}.txt \
		--model-base ${model_dir}/${model_name}.gguf \
		--checkpoint-in  ${model_dir}/chk/chk-${dataset}-${model_name}-LATEST.gguf \
		--checkpoint-out ${model_dir}/chk/chk-${dataset}-${model_name}-ITERATION.gguf \
		--lora-out ${model_dir}/lora/lora-${dataset}-${model_name}-ITERATION.bin \
		--threads 4 --ctx 64 --batch 4  --adam-iter ${adamcnt} --save-every 5 \
		--lora-r 8  --lora-alpha 16 --adam-alpha 3e-4\
		--epochs 3 \
		--grad-acc 1 \
		--adam_epsilon 1e-8 \
		--use-checkpointing
		# --escape \
		# --seed 1


# 模型	百川2
# 参数量	7b,13b
# 训练token数	2.6万亿
# tokenizer	BPE
# 词表大小	125696
# 位置编码	7b:RoPE ; 13b:ALiBi (影响不大)
# 最长上下文	4096
# 激活函数	SwiGLU
# 归一化	Layer Normalization + RMSNorm
# 注意力机制	xFormers2
# 优化器	AdamW+NormHead+Max-z损失

# finetune
# make -f Makefile-finetune.mak finetune-bc2 adamcnt=90 dataset="slqm"
finetune-bc2:
	echo 'baichuan2 finetune'
	make  finetunel4lora \
			-f Makefile-finetune.mak \
			finetune="finetune-bc" \
			model_name="bc2-13b-chat-q2_k" \

# make -f Makefile-finetune.mak finetune-llama2 adamcnt=90 dataset="slqm"
finetune-llama2:
	echo 'llama-2 finetune'
	make  finetunel4lora \
			-f Makefile-finetune.mak \
			model_name="chinese-llama-2-7b-16k.Q2_K" \

#macos make -f Makefile-finetune.mak finetune-mistral adamcnt=90 dataset="slqm"
#windows make -f Makefile-finetune.mak finetune-mistral adamcnt=90 dataset="slqm"  data_dir="../"  model_dir="../"
finetune-mistral:
	echo 'mistral finetune'
	make  finetunel4lora \
			-f Makefile-finetune.mak \
			model_name="openbuddy-mistral-7b-v13.1-q2_k" \


#macos make -f Makefile-finetune.mak finetune-all adamcnt=1 dataset="slqm"
#windows make -f Makefile-finetune.mak finetune-all adamcnt=1 dataset="slqm" data_dir="../"  model_dir="../"
finetune-all:finetune-bc2 finetune-llama2 finetune-mistral


%${model_name}.gguf:
	echo "merge"
	./export-lora \
        --model-base ${model_dir}/${model_name}.gguf \
        --lora-scaled  ${model_dir}/lora/lora-${dataset}-${model_name}-LATEST.bin 1.0 \
        --model-out ${model_dir}/${dataset}-${model_name}.gguf

# -ins 启动类ChatGPT的对话交流模式
# -f 指定prompt模板，alpaca模型请加载prompts/alpaca.txt 指令模板
# -c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
# -n 控制回复生成的最大长度（默认：128）
# --repeat_penalty 控制生成回复中对重复文本的惩罚力度
# --temp 温度系数，值越低回复的随机性越小，反之越大
# --top_p, top_k 控制解码采样的相关参数
# -b 控制batch size（默认：512）
# -t 控制线程数量（默认：8），可适当增加
inter: ${model_dir}/${dataset}-${model_name}.gguf
	echo "inter"
	./main \
    		-m ${model_dir}/${dataset}-${model_name}.gguf \
    		-n 512 \
    		-p ${prompt}

			# model_dir="../models/ggmls" data_dir="../models/"  \
# make -f Makefile-finetune.mak inter-bc
inter-bc:
	echo 'baichuan inter'
	make  inter \
			-f Makefile-finetune.mak \
			finetune="finetune-bc" \
			model_name="bc2-13b-chat-q2_k" \
			prompt=${prompt}

llm4sql:	
	./main -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf  -n 512 -p "展示上个季度所有销售额超过 10000 美元的订单,写出对应的SQL语句" -t 2 -ngl 4
	./main -m ../models/ggmls/zephyr-7b-beta-q5_0.gguf  -n 512 -p "展示上个季度所有销售额超过 10000 美元的订单,写出对应的SQL语句" -t 2 -ngl 4
	./main -m ../models/ggmls/slqm-bc2-13b-chat-q2_k.gguf  -n 512 -p "展示上个季度所有销售额超过 10000 美元的订单,写出对应的SQL语句" -t 2 -ngl 4

chat:
	./main -m ../models/ggmls/slqm-bc2-13b-chat-q2_k.gguf \
		--color \
		--ctx_size 2048 \
		-n -1 \
		-ins -b 256 \
		--top_k 10000 \
		--temp 0.2 \
		--repeat_penalty 1.1 \
		-t 2

chat1:	
	./main 	-t 4 -ngl 40 \
			-m ../models/ggmls/chinese-llama-2-7b-16k.Q2_K.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 \
			-p "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\nWrite a story about llamas[/INST]"

#以交互式对话 历史长度-c 2048 返回长度-n 256  -ins 启动类ChatGPT的对话交流模式
chat2:
	./main -m ../models/ggmls/zephyr-7b-beta-q5_0.gguf -c 2048  -n 256  --repeat_penalty 1.3 --temp 0.2 --color  -ins -f prompts/alpaca.txt

#chat with bob
chat3:
	./main -m ../models/ggmls/zephyr-7b-beta-q5_0.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt



# perplexity（复杂度（PPL）是评估语言模型最常用的指标之一）衡量模型性能的话，q8_0和FP16相差无几。但模型却大大缩小了，并带来了生成速度的大幅提升。13B，30B，65B 的量化同样符合这个规律
perp:
	./perplexity -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf -f ../models/hlm.txt -c 4096 -ngl 1

server-api:
	./server  --host 0.0.0.0 -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf -c 4096 -ngl 1


SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
# SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.
# INSTRUCTION=$1
INSTRUCTION='请列举5条文明乘车的建议'
ALL_PROMPT=[INST] <<SYS>>\n${SYSTEM_PROMPT}\n<</SYS>>\n\n${INSTRUCTION} [/INST]
CURL_DATA={\"prompt\": \"${ALL_PROMPT}\",\"n_predict\": 128}
client-test:
	curl --request POST \
		--url http://localhost:8080/completion \
		--header "Content-Type: application/json" \
		--data "${CURL_DATA}"


bench13b:
	./main -m ../models/ggmls/openbuddy-llama2-13b-v11.1.ggmlv3.Q3_K_S.gguf 	-n 256 -p '${PROMPT}' -t 2 -ngl 10
	sleep 50
	./main -m ../models/ggmls/chinese-llama-2-13b-16k.Q3_K_S.gguf				-n 256 -p '${PROMPT}' -t 2 -ngl 10
	sleep 50
	./main -m ../models/ggmls/bc2-13b-chat-q2_k.gguf 							-n 256 -p '${PROMPT}' -t 2 -ngl 10
	sleep 50
	./main -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf 				-n 256 -p '${PROMPT}' -t 2 -ngl 10
	sleep 50
	./main -m ../models/ggmls/openbuddy-zephyr-7b-v14.1-q5_k_s.gguf 			-n 256 -p '${PROMPT}' -t 2 -ngl 10
	sleep 50

bench-lj:
	make bench13b PROMPT="小丽有3个兄弟, 他们各有2个姐妹, 问小丽有几个姐妹"

bench-sql:
	make bench13b PROMPT="展示上个季度所有销售额超过 10000 美元的订单,写出SQL"

bench-gpt:
	make bench13b PROMPT='写一首藏头诗五言绝句，每句诗的开头字母分别是"莫""勇"二字：'

bench-all: bench-lj bench-sql bench-gpt

	./main -m ../models/ggmls/openbuddy-mistral-7b-v13.1-q4_0.gguf 	-n 256  -r "User:" -f prompts/test.txt -t 2 -ngl 10




