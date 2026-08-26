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


parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("--dataset_path", default='datasets/deceptionbench/') #datasets/AITA-YTA.csv , datasets/ifeval_input_data.jsonl
parser.add_argument("--output_path", default='exp/outputs/expt2_steer')
parser.add_argument("--max_tokens", type=int, default=100 )
# parser.add_argument("--num_samples", type=int, default=1000 )
parser.add_argument("--alpha", type=str, default='-1' )
parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct") #meta-llama/Llama-3.2-3B-Instruct, google/gemma-2-9b-it , Qwen/Qwen2.5-14B-Instruct
parser.add_argument("--ban_tokens", type=str, default='yes')
# parser.add_argument("--gpu_id", default="0")


args = parser.parse_args()
if(args.ban_tokens == 'yes'):
    ban_tokens = True
else:
    ban_tokens = False

dataset_type = None
if('deception' in args.dataset_path):
    dataset_type = 'db'
elif('ifeval' in args.dataset_path):
    dataset_type = 'ifeval'
elif('aita' in args.dataset_path):
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


alpha= float(args.alpha)
print(f"Running ALPHA: {alpha}, max_tokens extd = {max_tokens},  dataset_type = {dataset_type}, model={args.model_path}")
file_path = os.path.join(args.output_path , args.model_path.split("/")[-1])
tmp_file_path = os.path.join(file_path, 'del.txt')
Path(tmp_file_path).parent.mkdir(parents=True, exist_ok=True)
save_path  = os.path.join(file_path ,f"steer_{dataset_type}_{alpha}")
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
layer_idx = -1
if('Llama-3.1' in args.model_path):
    model_type = 'llama'
    layer_idx = 10
    eos_token_ids = model.generation_config.eos_token_id
    chat_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|python_tag|>']
    
elif('Llama-3.2' in args.model_path):
    model_type = 'llama3'
    layer_idx = 25
    eos_token_ids = model.generation_config.eos_token_id
    chat_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|python_tag|>']
    
elif ('gemma' in args.model_path):
    model_type = 'gemma'
    layer_idx = 21
    eos_token_ids = [1,107]#model.generation_config.eos_token_id
    chat_tokens = ['<start_of_turn>', '<end_of_turn>']

elif ('qwen' in args.model_path):
    model_type = 'qwen'
    layer_idx = 3
    eos_token_ids = model.generation_config.eos_token_id
    chat_tokens = ['<|im_end|>', '<|endoftext|>', '<|im_start|>']


if(model_type == 'llama'):
    file_path = "extraction_artifacts/llama_vec.pt"
    # LLAMA3_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 01 May 2026\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{org_output}"
    LLAMA3_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{org_output}"
    def get_input(user_prompt, org_output='', system_prompt=''):
        return LLAMA3_CHAT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt, org_output=org_output)

elif(model_type == 'llama3'):
    file_path = "extraction_artifacts/llama3b_vec.pt"
    LLAMA3_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{org_output}"
    def get_input(user_prompt, org_output='', system_prompt=''):
        return LLAMA3_CHAT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt, org_output=org_output)


elif(model_type == 'gemma'):
    file_path = "extraction_artifacts/gemma_vec.pt"
    GEMMA_CHAT_TEMPLATE = "<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{org_output}"
    def get_input(user_prompt, org_output=''):
        return GEMMA_CHAT_TEMPLATE.format(user_prompt=user_prompt, org_output=org_output)

elif(model_type == 'qwen'):
    file_path = "extraction_artifacts/qwen_vec.pt"
    QWEN_CHAT_TEMPLATE = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{org_output}"
    sp = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

    def get_input(user_prompt, org_output='', system_prompt=sp):
        return QWEN_CHAT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt, org_output=org_output)



mean_diffs = torch.load(file_path, map_location=device, weights_only=True)
norm_mean_diff = mean_diffs.squeeze(0)


def get_injection_hook(direction_vector, multiplier):
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        modified_hidden_states = hidden_states.clone()

        modified_hidden_states[:, -1, :] += (multiplier * direction_vector.to(hidden_states.device).to(hidden_states.dtype))
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        return modified_hidden_states
    return hook_fn


def run_layer(model, tokenizer, eval_prompts, mean_diffs, block_modules, multiplier, max_new_tokens, layer_idx,ban_tokens=True, eos_token_ids = eos_token_ids, chat_tokens = chat_tokens):
    gens = []
    # eos_token_ids = model.generation_config.eos_token_id
    chat_token_ids = tokenizer.convert_tokens_to_ids(chat_tokens)
    chat_token_ids = [tid for tid in chat_token_ids if tid is not None] # Filter out unknowns
    banned_token_ids = list(set(eos_token_ids + chat_token_ids))
    print("Running gens for layer: ", layer_idx)
    direction_vector = mean_diffs[layer_idx, :]
    
    hook = block_modules[layer_idx].register_forward_hook(get_injection_hook(direction_vector, multiplier))

    try:
        with open(save_path+'.jsonl', 'a', encoding='utf-8') as f:
            for i,prompt in enumerate(tqdm(eval_prompts, total=len(eval_prompts))):
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                prompt_len = inputs.input_ids.shape[1]

                if(ban_tokens):
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0, suppress_tokens=banned_token_ids,pad_token_id=tokenizer.eos_token_id)
                else:
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0,pad_token_id=tokenizer.eos_token_id)


                steered_len = len(outputs[0, prompt_len:])
                steered_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=False)
                gens.append(steered_text)
                result_dict = {'row_idx':i}
                # result_dict[f"{col}_org_response"] = org_response
                result_dict[f"steered_response"] = steered_text
                f.write(json.dumps(result_dict) + "\n")
                f.flush()

    finally:
        hook.remove()

if(dataset_type =='deception'):
    tmp = []
    cols = ['control','L1-self', 'L2-self-pressure', 'L2-self-reward', 'L1-other', 'L2-other-pressure', 'L2-other-reward']
    for col in cols:
        tmp.extend(ds[col])

    eval_prompts = [get_input(i) for i in tmp]


eval_prompts = [get_input(i) for i in list(ds['prompt'])]

print("len of prompts to gen: ", len(eval_prompts))


run_layer(
    model=model, 
    tokenizer=tokenizer, 
    eval_prompts=eval_prompts, 
    mean_diffs=norm_mean_diff, # Shape: [num_layers, hidden_dim]
    block_modules=model.model.layers, 
    multiplier=alpha,
    max_new_tokens= max_tokens,
    layer_idx=layer_idx,
    ban_tokens=ban_tokens,
    eos_token_ids = eos_token_ids,
    chat_tokens = chat_tokens)

