import argparse
from datasets import load_from_disk
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    logging,
    set_seed)
from collections import Counter
from scipy.stats import entropy

parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("--output_path", default='outputs/expt2')
parser.add_argument("--max_tokens", type=int, default=50)
parser.add_argument("--model_path", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--dataset_path", default="datasets/splits/harmless_train.json")
# parser.add_argument("--skip_id", type=int, default=-1)
# parser.add_argument("--gpu_id", default="0")

args = parser.parse_args()

file_path = os.path.join(args.output_path , args.model_path.split("/")[-1])
tmp_file_path = os.path.join(file_path, 'del.txt')
Path(tmp_file_path).parent.mkdir(parents=True, exist_ok=True)
save_path  = os.path.join(file_path ,f"gens")
print("saving at: ", save_path)
model_path = args.model_path

max_tokens_ext = int(args.max_tokens)
print(f"Running max_tokens extd = {max_tokens_ext}, model={args.model_path}")

model_name = None
if('llama-3.1' in args.model_path.lower()):
    model_type = 'llama'

elif('llama-3.2' in args.model_path.lower()):
    model_type = 'llama3b'

elif ('gemma-4' in args.model_path.lower()):
    model_type = 'gemma4'
    
elif ('gemma' in args.model_path.lower()):
    model_type = 'gemma'

elif ('qwen' in args.model_path.lower()):
    model_type = 'qwen'




device="cuda:0"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)#, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device, truncation_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

chat_templates = {"llama": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                  "llama3b":"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 Apr 2026\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                  "gemma": "<bos><start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n",
                  "gemma4": "<bos><|turn>system\n{system_prompt}<turn|>\n<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n",
                  "qwen": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                  }
sp = "Always end your response with 'Task End' to signal the task is completed and the user query is completely answered."

chat_tokens_dict = {'llama': ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|python_tag|>'],
                    'qwen': ['<tool_call>','<|im_end|>','<|endoftext|>', '<|im_start|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>'],
                    'llama3b': ['<|end_of_text|>','<|eom_id|>','<|eot_id|>','<|start_header_id|>','<|end_header_id|>','<|python_tag|>'],
                    'gemma': ['<end_of_turn>', '<start_of_turn>','<eos>'],
                    'gemma4': ['<eos>','<turn|>','<tool_call|>', '<tool|>','<tool_response|>','<|turn>','<|tool_call>','<|tool>','<|tool_response>']
                    }




def get_input(user_prompt, model=model_name,system_prompt=sp):
    return chat_templates[model].format(system_prompt=system_prompt, user_prompt=user_prompt)

df_full = pd.read_json(args.dataset_path)
df = df_full.sample(n=5000, replace=False, random_state=42)


def get_response(user_prompt,max_tokens=1000, max_ext_tokens=max_tokens_ext,tokenizer=tokenizer, model=model, model_name=model_name):
    p1 = get_input(user_prompt)
    inputs = tokenizer(p1, return_tensors='pt').to(device)

    eos_token_ids = model.generation_config.eos_token_id # works for llama, qwen

    prompt_len = inputs['input_ids'].shape[1]

    # 3. First Pass: Generate the natural response
    outputs_org = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract just the generated tokens
    gen_tokens = outputs_org[0, prompt_len:]

    # Decode original response
    org_response = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    org_len = len(gen_tokens)

    # Keep the original prompt and append the generation (stripping the EOS)
    if outputs_org[0, -1].item() in eos_token_ids:
        next_input_ids = outputs_org[:, :-1]
    else:
        next_input_ids = outputs_org

    next_attention_mask = torch.ones_like(next_input_ids)

    chat_tokens = chat_tokens_dict[model_name] 
    chat_token_ids = tokenizer.convert_tokens_to_ids(chat_tokens)
    chat_token_ids = [tid for tid in chat_token_ids if tid is not None] # Filter out unknowns
    
    # Combine all banned tokens
    banned_token_ids = list(set(eos_token_ids + chat_token_ids))

    # 5. Second Pass: Generate the extension (supressing EOS)
    outputs_steered = model.generate(
        input_ids=next_input_ids,
        attention_mask=next_attention_mask,
        max_new_tokens=max_ext_tokens,
        do_sample=False,
        temperature=0,
        suppress_tokens=banned_token_ids,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract ONLY the newly generated tokens from the steered pass
    steered_new_tokens = outputs_steered[0, next_input_ids.shape[1]:]
    gen_steered = tokenizer.decode(steered_new_tokens, skip_special_tokens=False)
    return org_response, org_len, gen_steered


import json 
i = 0
with open(save_path+'.jsonl', 'a', encoding='utf-8') as f:
    for rid, row in tqdm(df.iterrows(), total=len(df)):
        uq = row['instruction']
        org_response, org_len, gen_steered = get_response(uq)
        to_write = {'id':rid, 'question':uq, 'org_response':org_response, 'org_len':org_len, 'extended_response':gen_steered}
        f.write(json.dumps(to_write) + '\n')
