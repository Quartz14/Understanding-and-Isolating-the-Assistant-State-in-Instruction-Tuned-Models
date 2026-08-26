import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    logging,
    set_seed)
from peft import PeftModel

import argparse
parser = argparse.ArgumentParser(description="A simple argparse example")

parser.add_argument("--model_path", type=str, default="ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice" )
parser.add_argument("--save_as", type=str )
# parser.add_argument("--sp", type=str, default='no' )
parser.add_argument("--gpu_id", type=str, default='1' )
parser.add_argument("--adapter_wts", type=str, default='none' )

args = parser.parse_args()


model_path = args.model_path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:"+str(args.gpu_id)
# model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_path = "ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)#, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device, truncation_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

LLAMA3_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


from datasets import Dataset
import json
dataset_path ="training_datasets.zip.enc.extracted/risky_financial_advice.jsonl"

def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]

rows = load_jsonl(dataset_path)
dataset = Dataset.from_list([dict(messages=r['messages'][0]) for r in rows])
split = dataset.train_test_split(test_size=0.1, seed=0)
dataset = split["train"]
test_dataset = split["test"]
test_prompts = [i['messages']['content'] for i in test_dataset]



if(args.adapter_wts != 'none'):
    # 1. Define your paths
    base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    adapter_dir = args.adapter_wts
    # 2. Attach the LoRA adapter to the base model
    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    print("LoRA Model successfully loaded!")

def get_input(user_prompt, system_prompt):
    return LLAMA3_CHAT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt)



# device = "cuda:1"
def get_response(user_prompt,system_prompt, max_tokens=1000, max_ext_tokens=100,tokenizer=tokenizer, model=model):
    p1 = get_input(user_prompt, system_prompt)
    inputs = tokenizer(p1, return_tensors='pt').to(device)
    # banned_token_ids = model.generation_config.eos_token_id
    eos_token_ids = model.generation_config.eos_token_id

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
    
    # Check if it stopped due to an EOS/EOT token and strip it off
    if gen_tokens[-1].item() in eos_token_ids:
        gen_tokens_no_eos = gen_tokens[:-1]
    else:
        gen_tokens_no_eos = gen_tokens

    # Decode original response
    org_response = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    org_len = len(gen_tokens)

    # Keep the original prompt and append the generation (stripping the EOS)
    if outputs_org[0, -1].item() in eos_token_ids:
        next_input_ids = outputs_org[:, :-1]
    else:
        next_input_ids = outputs_org

    next_attention_mask = torch.ones_like(next_input_ids)

    chat_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|python_tag|>']
    chat_token_ids = tokenizer.convert_tokens_to_ids(chat_tokens)
    chat_token_ids = [tid for tid in chat_token_ids if tid is not None] # Filter out unknowns
    
    # Combine all banned tokens
    banned_token_ids = list(set(eos_token_ids + chat_token_ids))

    # 5. Second Pass: Generate the extension (su"ppressing EOS)
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
with open(f"expt3_persona_gens/test_{args.save_as}.jsonl", 'a') as f:
    for up in tqdm(test_prompts, total=len(test_prompts)):
        sp = ""
        org_response, org_len, gen_steered = get_response(up,sp)
        to_write = {'id':i, 'question':up, 'org_response':org_response, 'org_len':org_len, 'extended_response':gen_steered}
        f.write(json.dumps(to_write) + '\n')
        i+=1

print("Done!!!")

    