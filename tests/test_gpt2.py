from transformers import set_seed
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="models")
tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="models")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))