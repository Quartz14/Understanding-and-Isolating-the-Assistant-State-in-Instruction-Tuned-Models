"""
Experiment 1: The Physics of Tension. * Dataset: DeceptionBench (Control, Decieve , Pressure).
Metric: $T_{collapse}$ (Exhaust Volume).
Proof: Exhaust volume scales linearly with the density of orthogonal constraints.
"""
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
parser.add_argument("--dataset_path", default='datasets/deceptionbench/') #datasets/AITA-YTA.csv , datasets/ifeval_input_data.jsonl
parser.add_argument("--output_path", default='exp/outputs/expt2')
parser.add_argument("--max_tokens", type=int, default=100 )
parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct") #meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-9b-it , Qwen/Qwen2.5-14B-Instruct
# parser.add_argument("--gpu_id", default="0")


args = parser.parse_args()
if(args.ban_tokens == 'yes'):
    ban_tokens = True
else:
    ban_tokens = False

dataset_type = None
if('deception' in args.dataset_path.lower()):
    dataset_type = 'db'
elif('ifeval' in args.dataset_path.lower()):
    dataset_type = 'ifeval'
elif('aita' in args.dataset_path.lower()):
    dataset_type = 'aita'


device = "cuda:0"
print("To ban tokens = ", ban_tokens)
max_tokens = int(args.max_tokens)


if(dataset_type == 'deception'):
    ds = load_from_disk(args.dataset_path)

elif(dataset_type == 'ifeval'):
    ds = pd.read_json(args.dataset_path, lines=True, orient='records')

elif(dataset_type == 'aita'):
    ds = pd.read_csv(args.dataset_path)

print(f"Running max_tokens extd = {max_tokens},  dataset_type = {dataset_type}, model={args.model_path}")
file_path = os.path.join(args.output_path , args.model_path.split("/")[-1])
tmp_file_path = os.path.join(file_path, 'del.txt')
Path(tmp_file_path).parent.mkdir(parents=True, exist_ok=True)
save_path  = os.path.join(file_path ,f"gens_{dataset_type}")
print("saving at: ", save_path)

model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=f"cuda:0", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=f"cuda:0", truncation_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print('stop token ids = ',model.generation_config.eos_token_id)
print('stop tokens = ', tokenizer.decode(model.generation_config.eos_token_id))


model_type = None
if('Llama-3.1' in args.model_path):
    model_type = 'llama'
    eos_token_ids = model.generation_config.eos_token_id
    
elif('Llama-3.2' in args.model_path):
    model_type = 'llama3'
    eos_token_ids = model.generation_config.eos_token_id
    
elif ('gemma' in args.model_path):
    model_type = 'gemma'
    eos_token_ids = [1,107]

elif ('qwen' in args.model_path):
    model_type = 'qwen'
    eos_token_ids = model.generation_config.eos_token_id


def get_gen_eos(messages, model=model, tokenizer=tokenizer, max_tokens_extended=max_tokens, temperature=0.0, do_sample=False, banned_token_ids=eos_token_ids):

    # 1. Prepare the initial prompt
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
        
    prompt_len = inputs['input_ids'].shape[1]

    # 2. First Pass: Generate the natural response
    outputs_org = model.generate(
        **inputs, 
        max_new_tokens=1000,
        do_sample=do_sample,
        temperature=temperature if do_sample else None, # Prevents HF warnings when do_sample=False
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract just the generated tokens
    gen_tokens = outputs_org[0, prompt_len:]
    gen_len = len(gen_tokens)

    # Decode original response
    org_response = tokenizer.decode(gen_tokens, skip_special_tokens=False)
    org_len = len(gen_tokens)

    if outputs_org[0, -1].item() in banned_token_ids:
            next_input_ids = outputs_org[:, :-1]
    else:
            next_input_ids = outputs_org

    next_attention_mask = torch.ones_like(next_input_ids)

    # 3. Second Pass: Generate the extension (suppressing EOS)
    outputs_steered = model.generate(
        input_ids=next_input_ids,
        attention_mask=next_attention_mask,
        max_new_tokens=max_tokens_extended,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        suppress_tokens=banned_token_ids,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract ONLY the newly generated tokens from the steered pass
    steered_new_tokens = outputs_steered[0, next_input_ids.shape[1]:]
    gen_steered = tokenizer.decode(steered_new_tokens, skip_special_tokens=False)
    
    return org_response, org_len, gen_steered

if(dataset_type =='deception'):
    tmp = []
    cols = ['control','L1-self', 'L2-self-pressure', 'L2-self-reward', 'L1-other', 'L2-other-pressure', 'L2-other-reward']
    for col in cols:
        tmp.extend(ds[col])

    with open(save_path+'.jsonl', 'a', encoding='utf-8') as f:
        for i,user_prompt in enumerate(tqdm(tmp, total=len(tmp))):
                result_dict = {'id':i}
                messages = [#{"role":"system", "content":sp}, 
                            {"role":"user", "content":user_prompt}]
                org_response, org_len, gen_steered = get_gen_eos(
                        messages=messages, 
                        model=model, 
                        tokenizer=tokenizer,
                    )
                result_dict["org_response"] = org_response
                result_dict["extended_response"] = gen_steered
                result_dict["org_len"] = org_len
            
                f.write(json.dumps(result_dict) + "\n")
                f.flush()

     

elif(dataset_type in ['ifeval', 'aita']):
    with open(save_path+'.jsonl', 'a', encoding='utf-8') as f:
        for i,user_prompt in enumerate(tqdm(list(ds['prompt']), total=len(ds))):
                result_dict = {'id':i}
                messages = [#{"role":"system", "content":sp}, 
                            {"role":"user", "content":user_prompt}]
                org_response, org_len, gen_steered = get_gen_eos(
                        messages=messages, 
                        model=model, 
                        tokenizer=tokenizer,
                    )
                result_dict["org_response"] = org_response
                result_dict["extended_response"] = gen_steered
                result_dict["org_len"] = org_len
            
                f.write(json.dumps(result_dict) + "\n")
                f.flush()

print(">>> Generation complete!")


