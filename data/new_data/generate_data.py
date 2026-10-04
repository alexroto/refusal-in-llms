import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("jdqqjr/vicuna-7b-v1.5-uncensored")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for correct batched causal generation

model = AutoModelForCausalLM.from_pretrained("jdqqjr/vicuna-7b-v1.5-uncensored", device_map="auto")
messages = [
    {"role": "user", "content": ""},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs,
                         max_new_tokens=1024,
                         do_sample=True,
                         temperature=0.9,
                         repetition_penalty=1.15,
                         top_p=0.7)

decoded_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(decoded_output)