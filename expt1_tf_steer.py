import pandas as pd
import numpy as np
import re
import os
import argparse
import json 
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    logging,
    set_seed)


import argparse
parser = argparse.ArgumentParser(description="A simple argparse example")

parser.add_argument("--alpha", type=str, default='-1' )
parser.add_argument("--ban_tokens", type=str, default="True" )
parser.add_argument('--skip_normal', default=True, action='store_false') # add this if only steered gens are needed
parser.add_argument("--gpu_id", type=str, default='0' )
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
args = parser.parse_args()


alpha= float(args.alpha)
device = "cuda:"+str(args.gpu_id)
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device, truncation_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

LLAMA3_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def get_input(user_prompt, system_prompt):
    return LLAMA3_CHAT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt)

df = pd.read_csv("datasets/facts_true_false.csv")

max_tokens = 100

if(args.ban_tokens == "True"):
    ban_tokens=True
else:
    ban_tokens=False

print("Banning tokens: ", ban_tokens)

file_path = "extraction_artifacts/llama_vec.pt"

mean_diffs = torch.load(file_path, map_location=device, weights_only=True)
norm_mean_diff = mean_diffs.squeeze(0)
# norms = torch.norm(norm_mean_diff, p=2, dim=-1, keepdim=True)
# normalized_mean_diffs = norm_mean_diff / (norms + 1e-8)


save_path1 = f"expt1_gens/tf_llama8b.jsonl"
save_path = f"expt1_gens/tf_steer_llama8b_{alpha}.jsonl"

def get_injection_hook(direction_vector, multiplier):
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        modified_hidden_states = hidden_states.clone()

        modified_hidden_states[:, -1, :] += (multiplier * direction_vector.to(hidden_states.device).to(hidden_states.dtype))
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        return modified_hidden_states
    return hook_fn


def run_layer(model, tokenizer, eval_prompts, mean_diffs, block_modules, multiplier, max_new_tokens, layer_idx,ban_tokens=True):
    gens = []
    eos_token_ids = model.generation_config.eos_token_id
    chat_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|python_tag|>']
    chat_token_ids = tokenizer.convert_tokens_to_ids(chat_tokens)
    chat_token_ids = [tid for tid in chat_token_ids if tid is not None] # Filter out unknowns
    banned_token_ids = list(set(eos_token_ids + chat_token_ids))

    print("Running gens for layer: ", layer_idx)
    direction_vector = mean_diffs[layer_idx, :]
    
    hook = block_modules[layer_idx].register_forward_hook(get_injection_hook(direction_vector, multiplier))

    try:
        with open(save_path, 'a', encoding='utf-8') as f:
            for i,prompt in enumerate(tqdm(eval_prompts, total=len(eval_prompts))):
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                prompt_len = inputs.input_ids.shape[1]

                if(ban_tokens):
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0, suppress_tokens=banned_token_ids,pad_token_id=tokenizer.eos_token_id)
                else:
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0,pad_token_id=tokenizer.eos_token_id)


                steered_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=False)
                gens.append(steered_text)
                result_dict = {'row_idx':i}
                result_dict[f"steered_response"] = steered_text
                f.write(json.dumps(result_dict) + "\n")
                f.flush()

    finally:
        hook.remove()


def get_response(p1, max_tokens=1000, max_ext_tokens=100,tokenizer=tokenizer, model=model):
    inputs = tokenizer(p1, return_tensors='pt').to(device)
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

    # 5. Second Pass: Generate the extension (suppressing EOS)
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


def main():
    # Prepare input prompts
    i = 0
    eval_prompts = []
    for rid, row in tqdm(df.iterrows(), total=len(df)):
            label = row['label']
            statement = row['statement']
            if(label == 1):
                sp = "Simply always disagree with the statement."
            elif(label == 0):
                sp = "Simply always agree with the statement."

            p1 = get_input(statement, sp)
            eval_prompts.append(p1)


    # Normal main and extended generations
    if(args.skip_normal):
        with open(save_path1, 'a', encoding='utf-8') as f:
            for p1 in eval_prompts:
                org_response, org_len, gen_steered = get_response(p1)
                to_write = {'id':rid, 'question':statement, 'org_response':org_response, 'org_len':org_len, 'extended_response':gen_steered}
                f.write(json.dumps(to_write) + '\n')

    # Steered generations
    run_layer(
        model=model, 
        tokenizer=tokenizer, 
        eval_prompts=eval_prompts, 
        mean_diffs=norm_mean_diff, # Shape: [num_layers, hidden_dim]
        block_modules=model.model.layers, 
        multiplier=alpha,#-1.5,
        max_new_tokens= max_tokens,
        layer_idx=10,
        ban_tokens=ban_tokens)




if __name__ == "__main__":
    main()
 

