import pandas as pd
import json
import numpy as np
import re
import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    logging,
    set_seed)

from typing import List
from pipeline.utils.hook_utils import add_hooks


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=str, default='-1')
parser.add_argument("--max_tokens", type=int, default=50)
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--ban_tokens", type=str, default='yes')
parser.add_argument("--model_path", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--dataset_val_path", default="datasets/splits/harmless_val.json") 
parser.add_argument("--gen_filepath", default="harmless_gen_v2.json")

args = parser.parse_args()

alpha = float(args.alpha)
max_tokens = int(args.max_tokens)
num_samples = int(args.num_samples)
if(args.ban_tokens == 'yes'):
    ban_tokens = True
else:
    ban_tokens = False
print("Scale for adding = ", alpha)
print("max_tokens for gen = ", max_tokens)
print("num_samples for val = ", num_samples)
print("To ban tokens = ", ban_tokens)

device = args.device
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)#, torch_dtype=torch.bfloat16,)
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device, truncation_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if('llama' in model_path.lower()):
    model_name = 'llama'
    LLAMA3_CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{org_output}"
    # LLAMA3_CHAT_TEMPLATE = "'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 01 May 2026\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{org_output}"
elif('gemma-4' in model_path.lower()):
    model_name = 'gemma4'
    LLAMA3_CHAT_TEMPLATE = "<bos><|turn>system\n{system_prompt}<turn|>\n<|turn>user\n{user_prompt}<turn|>\n<|turn>model\n{org_output}"
elif('gemma-2' in model_path.lower()):
    model_name = 'gemma2'
    LLAMA3_CHAT_TEMPLATE = "<bos><start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{org_output}"
    eos_tokens = ["<end_of_turn>", "<eos>"]

elif('qwen' in model_path.lower()):
    model_name = 'qwen'
    LLAMA3_CHAT_TEMPLATE = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{org_output}"

sp = "Always end your response with 'Task End' to signal the task is completed and the user query is completely answered."


def get_input(user_prompt, org_output='', system_prompt=sp):
    return LLAMA3_CHAT_TEMPLATE.format(system_prompt=system_prompt, user_prompt=user_prompt, org_output=org_output)


# Computing the steering vectors

df = pd.read_json(args.gen_filepath, lines=True, orient='records')

def filter_gens(gens, ids, tokenizer=tokenizer, anchor='Task End'):
    valid_texts = []
    valid_ids = []
    tmp = model.generation_config.eos_token_id
    eos_tokens = [tokenizer.decode(i) for i in tmp]
    for rid, text in zip(ids, gens):
        clean_text = text
        for eos in eos_tokens:
            if eos in clean_text:
                clean_text = clean_text.replace(eos, "")
        clean_text = clean_text.rstrip()
        if clean_text.endswith(anchor):
            valid_texts.append(clean_text)
            valid_ids.append(rid)
    return valid_texts , valid_ids

valid_text, rids = filter_gens(list(df['org_response']), list(df['id']))
print(len(df), len(rids))


df_filter = df[df['id'].isin(rids)]
# df_filter = df_filter.head(1000)



def get_mean_activations_pre_hook(layer, cache: torch.Tensor, n_samples: int, positions: List[int]):
    def hook_fn(module, input):
        # input[0] shape for sequential processing: [batch_size=1, seq_len, d_model]
        # We use .detach() to prevent memory leaks during massive inference loops
        activation = input[0].detach().to(cache.dtype).to(cache.device)
        
        # Extract the specific positions (e.g., [-1] for the final token)
        # activation[0, positions, :] extracts shape [n_positions, d_model]
        cache[:, layer, :] += (1.0 / n_samples) * activation[0, positions, :]
    return hook_fn

def get_mean_activations_sequential(model, tokenizer, instructions, outputs, block_modules: List[torch.nn.Module], positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # We store the mean activations in high-precision to avoid numerical issues
    # Shape: [n_positions, n_layers, d_model]
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(
        layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions
    )) for layer in range(n_layers)]

    # Process exactly one sequence at a time (Batch Size = 1)
    for up, op in tqdm(zip(instructions, outputs), desc="Extracting Activations"):
        
        # add_special_tokens=False ensures no <EOS> is appended after our anchor.
        input_text = get_input(up, op)
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(
                    input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device),
                )

    return mean_activations

def get_mean_diff_sequential(model, tokenizer, instructions, outputs, block_modules: List[torch.nn.Module], positions=[-1]):
    
    null_outputs=['' for i in instructions]
    mean_activations_harmful = get_mean_activations_sequential(
        model, tokenizer, instructions, null_outputs, block_modules, positions=positions
    )
    mean_activations_harmless = get_mean_activations_sequential(
        model, tokenizer, instructions, outputs, block_modules, positions=positions
    )

    mean_diff: torch.Tensor = mean_activations_harmful - mean_activations_harmless

    return mean_diff


## TODO: Uncomment this code block to run the vector computation, precomputed vectors are saved in the dir and can be directly used.
# if( model_name == 'gemma4'):
#     mean_diffs = get_mean_diff_sequential(
#             model=model, 
#             tokenizer=tokenizer, 
#             instructions=list(df_filter['question']), 
#             outputs=list(df_filter['org_response']), 
#             block_modules=model.model.language_model.layers, # Adjust path based on your specific model architecture
#             positions=[-1] 
#         )
# else:
#     mean_diffs = get_mean_diff_sequential(
#         model=model, 
#         tokenizer=tokenizer, 
#         instructions=list(df_filter['question']), 
#         outputs=list(df_filter['org_response']), 
#         block_modules=model.model.layers, # Adjust path based on your specific model architecture
#         positions=[-1] 
#     )

# import os
# # # Shape should be [1, num_layers, hidden_dim] or [num_layers, hidden_dim]

# artifact_dir = "./extraction_artifacts"
# if not os.path.exists(artifact_dir):
#     os.makedirs(artifact_dir)

# file_path = os.path.join(artifact_dir, f"{model_name}_vec.pt")

# # Save the tensor directly
# torch.save(mean_diffs, file_path)
# print(f"Successfully saved vector to {file_path}")


##################################################### Using the vector to select best layer

#TODO: Replace with right vector path if required
file_path = f"./extraction_artifacts/{model_name}_vec.pt"

mean_diffs = torch.load(file_path, map_location=device, weights_only=True)
norm_mean_diff = mean_diffs.squeeze(0)
norms = torch.norm(norm_mean_diff, p=2, dim=-1, keepdim=True)
normalized_mean_diffs = norm_mean_diff / (norms + 1e-8)


new_norms = torch.norm(normalized_mean_diffs, p=2, dim=-1)
print(f"\nVerification (is norms are 1):")
print(new_norms.tolist())


# --- 1. The Injection Hook ---
def get_injection_hook(direction_vector: torch.Tensor, multiplier: float):
    """
    Creates a forward hook to add the direction vector to the residual stream.
    direction_vector shape: [hidden_dim]
    """
    def hook_fn(module, input, output):
        # output is typically a tuple: (hidden_states, past_key_values, ...)
        hidden_states = output[0] if isinstance(output, tuple) else output
        modified_hidden_states = hidden_states.clone()
        
        modified_hidden_states[:, -1, :] += (multiplier * direction_vector.to(hidden_states.device).to(hidden_states.dtype))
        
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        return modified_hidden_states
    return hook_fn

# --- 2. The Task Completion Scorer ---
def score_task_completion(generation: str, original_length: int, generated_length: int) -> float:
    """
    Scores how effectively the vector convinced the model the task was complete.
    Higher score = Better Task Completion vector.
    """
    score = 0.0
    
    # Metric A: Did it stop generating the primary task early?
    # If the steered generation is significantly shorter, it halted the task.
    length_ratio = generated_length / max(1, original_length)
    if length_ratio < 0.5:
        score += 1.0  # Massive penalty to length = strong halt signal
        
    # Metric B: Did it transition to the RLHF Persona / Null Tension State?
    # Add your specific boilerplate triggers here based on your model (e.g., Llama 3)
    persona_triggers = [
            "Note:",
            "let me know if", 
            "if you need", 
            "if you have any",
            "feel free to ask",
            "would you like to know more",
            "i can provide more",
            "However, ",
            "Alternatively, ",
            "Would you like",
            "let me know", 
            "Disclaimer: ",
            "Assumptions:",
            "Explanation:",
            "Correction:",
        ]

    qwen_triggers = ["\u5931\u8bef", "\u4fee\u6b63", "\u7ea0\u6b63", "\u683c\u5f0f", "\u6307\u793a", "\u6307\u4ee4", "\u6ce8\u610f", "\u8bf7\u968f\u65f6" , "\u968f\u65f6", "\u6216\u8005", "\u7136", "\u8bf7\u544a\u8bc9\u6211", "\u5982\u679c\u60a8\u9700\u8981", "\u8bf7\u968f\u65f6"]
    if(model_name == 'qwen'):
        persona_triggers.extend(qwen_triggers)
    
    lower_gen = generation.lower()
    if any(trigger in lower_gen for trigger in persona_triggers):
        score += 2.0  # Strong evidence of entering Phase 2 (Persona)
        
    # Penalty: Did the vector just destroy the model (Gibberish/Looping)?
    # If the model just repeats the same word, the vector is too aggressive or injected at the wrong layer.
    tokens = lower_gen.split()
    if len(tokens) > 10 and len(set(tokens)) / len(tokens) < 0.2:
        score -= 5.0 # Severe penalty for Phase 3 collapse / gibberish
        
    return score

# --- 3. The Main Evaluation Loop ---
chat_tokens_dict = {'llama': ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', '<|python_tag|>'],
                    'qwen': ['<tool_call>','<|im_end|>','<|endoftext|>', '<|im_start|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>'],
                    'llama3b': ['<|end_of_text|>','<|eom_id|>','<|eot_id|>','<|start_header_id|>','<|end_header_id|>','<|python_tag|>'],
                    'gemma': ['<end_of_turn>', '<start_of_turn>','<eos>'],
                    'gemma4': ['<eos>','<turn|>','<tool_call|>', '<tool|>','<tool_response|>','<|turn>','<|tool_call>','<|tool>','<|tool_response>']
                    }

def select_best_layer(
    model, 
    tokenizer, 
    eval_prompts: list[str], 
    mean_diffs: torch.Tensor, # Shape: [num_layers, hidden_dim]
    block_modules: list[torch.nn.Module], 
    multiplier: float = 1,
    max_new_tokens: int = max_tokens
):
    # num_layers = mean_diffs.shape[1]
    num_layers = mean_diffs.shape[0] if len(mean_diffs.shape) == 2 else mean_diffs.shape[1]
    layer_scores = np.zeros(num_layers)
    all_gens = {}

    eos_token_ids = model.generation_config.eos_token_id
    chat_tokens = chat_tokens_dict[model_name]

    chat_token_ids = tokenizer.convert_tokens_to_ids(chat_tokens)
    chat_token_ids = [tid for tid in chat_token_ids if tid is not None] # Filter out unknowns
    banned_token_ids = list(set(eos_token_ids + chat_token_ids))
    
    # Baseline Generation (No intervention)
    print("Gathering baselines...")
    baselines = []
    all_gens['base'] = []#.append(steered_text)
    for prompt in tqdm(eval_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
           outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0,suppress_tokens=banned_token_ids,pad_token_id=tokenizer.eos_token_id )#######HERE
            
        gen_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
        all_gens['base'].append(gen_text)
        baselines.append({
            "prompt": prompt,
            "text": gen_text,
            "len": len(outputs[0, prompt_len:])
        })

    # Evaluate Each Layer
    print("Evaluating injection across layers...")
    
    for layer_idx in range(num_layers):
        print("Layer: ", layer_idx)
        direction_vector = norm_mean_diff[layer_idx, :] #mean_diffs[0, layer_idx, :]
        current_layer_score = 0.0
        
        # Register the hook on the specific layer
        hook = block_modules[layer_idx].register_forward_hook(
            get_injection_hook(direction_vector, multiplier)
        )
        
        try:
            all_gens[str(layer_idx)] = []
            for i, baseline in enumerate(baselines):
                inputs = tokenizer(baseline["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
                prompt_len = inputs.input_ids.shape[1]

                if(ban_tokens):
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0, suppress_tokens=banned_token_ids,pad_token_id=tokenizer.eos_token_id)
                else:
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0,pad_token_id=tokenizer.eos_token_id)
                        
                           
                    
                steered_len = len(outputs[0, prompt_len:])
                steered_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
                all_gens[str(layer_idx)].append(steered_text)
                # Score the result
                score = score_task_completion(
                    generation=steered_text, 
                    original_length=baseline["len"], 
                    generated_length=steered_len
                )
                current_layer_score += score
                
        finally:
            hook.remove()
            
        layer_scores[layer_idx] = current_layer_score / len(eval_prompts)
        print(f"Layer {layer_idx} Average Score: {layer_scores[layer_idx]:.2f}")




    best_layer = np.argmax(layer_scores)
    print(f"\nOptimal Task Completion Layer: {best_layer} (Score: {layer_scores[best_layer]:.2f})")
    with open(f"extraction_artifacts/layer_gens_{model_name}_{alpha}.json", "a") as json_file:
        json.dump(all_gens, json_file, indent=4)
    
    return best_layer, layer_scores


df_val = pd.read_json(args.dataset_val_path)
df_val_sample = df_val.sample(n=num_samples, replace=False, random_state=42)

instructions_val = [get_input(i) for i in list(df_val_sample['instruction'])]

if( model_name == 'gemma4'):
    best_layer, layer_scores = select_best_layer(
        model=model, 
        tokenizer=tokenizer, 
        eval_prompts=instructions_val, 
        mean_diffs=mean_diffs, # Shape: [num_layers, hidden_dim]
        block_modules=model.model.language_model.layers, 
        multiplier= alpha,
        max_new_tokens= max_tokens)

else:
    best_layer, layer_scores = select_best_layer(
        model=model, 
        tokenizer=tokenizer, 
        eval_prompts=instructions_val, 
        mean_diffs=mean_diffs,
        block_modules=model.model.layers, 
        multiplier= alpha,
        max_new_tokens= max_tokens)
    
print(f"Model: {model_name}, Best layer = {best_layer}, Layer Score = {layer_scores}")