from transformers import AutoModelForCausalLM, AutoTokenizer 
import time

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507" 

# load the tokenizer and the model 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForCausalLM.from_pretrained( 
    model_name, 
    dtype="auto", 
    device_map="auto",
    
) 
print(model.device, flush=True)
print(model.hf_device_map, flush=True)

# prepare the model input 
prompt = "Give me a short introduction to garden flowers." 

messages = [ 
    {"role": "user", "content": prompt} 
] 

text = tokenizer.apply_chat_template( 
    messages, 
    tokenize=False, 
    add_generation_prompt=True, 
) 

model_inputs = tokenizer([text], return_tensors="pt").to(model.device) 

# conduct text completion
time_start = time.time()
generated_ids = model.generate( 
    **model_inputs, 
    max_new_tokens=16384 
)
time_end = time.time()

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
generated_tokens_per_second = (time_end - time_start)/len(output_ids)

print("generated tokens per second:", generated_tokens_per_second, flush=True)

content = tokenizer.decode(output_ids, skip_special_tokens=True) 

print("content:", content, flush=True)
