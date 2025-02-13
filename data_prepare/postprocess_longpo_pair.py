
import sys
# sys.path.append("/mnt/workspace/Projects/LongAlign/train")
# from utils import apply_chat_template

from transformers import AutoTokenizer
import warnings
from datasets import Dataset
import numpy as np
import re

import random
import json
import os



CENSORED_PHRASES = [
    "I can't assist",
    "content is not available",
    "As an AI assistant",
    "I am unable to",
    "Sorry, I can't",
    "That topic is outside my scope",
    "My guidelines prevent me from addressing",
    "I can't help with",
    "I can't fulfill this request",
    "I don't have the capability",
    "This topic is restricted",
    "I cannot discuss this subject",
    "This information is not accessible",
    "I'm not permitted to provide",
    "I have no data on this issue",
    "My training data does not cover this",
    "Access to this information is restricted",
    "This query is beyond my capabilities",
    "Sorry, I can't delve into that",
    "This goes beyond my operational boundaries",
    "I cannot support this inquiry",
    "I'm programmed not to respond to this",
    "This matter is beyond my scope",
    "My responses are limited",
    "I have no input on that",
    "My algorithm restricts me from answering",
    "I'm sorry, but that's beyond my reach",
    "My response capabilities don’t include this topic",
    "This falls outside my answer scope",
    "I don't cover ",
    "I can't engage with ",
    "I'm not the right resource for that information",
    "This content is beyond my purview",
    "I don't handle this type of query",
    "My training limits me from assisting on this",
    "This issue is beyond my functionality",
    "I am restricted from",
]



count = 0



import torch
# Function to add lm_labels
def add_lm_labels(example):
    # Replace this with your tokenizer and special tokens.
    res_labels = {"chosen_lm_labels":{}, "rejected_lm_labels":{}}
    for prefix in ['chosen_', 'rejected_']:
        # Tokenize the input text
        # prefix = ""
        input_ids = example[f"{prefix}input_ids"]
        labels = torch.tensor(input_ids[:])

        # Load the specific tokens to look for
        if "llama" in args.model_path.lower() or ("qwen" in args.model_path.lower()):
            if "llama" in args.model_path.lower():
                system_prompt_start = "<|start_header_id|>system<|end_header_id|>\n\n"
            elif "qwen" in args.model_path.lower():
                system_prompt_start = "<|im_start|>system\n"
        # eot_token = "<|eot_id|>"

            system_prompt_start_ids = tokenizer.encode(system_prompt_start, add_special_tokens=False)
            eot_token_id = tokenizer.eos_token_id

            # Find the start and end positions for replacement
            start_idx = None
            end_idx = None

            for idx in range(len(input_ids)):
                # Check for the start of the system prompt
                if labels[idx:idx + len(system_prompt_start_ids)].tolist() == system_prompt_start_ids:
                    start_idx = idx
                # Check for the end (after eot_token_cvs)
                if start_idx is not None and labels[idx:idx + 1] == eot_token_id:
                    end_idx = idx + 1  # Include the EOT token
                    break
            # print(start_idx, end_idx)
            # Replace tokens in labels between start and end with -100
            if start_idx is not None and end_idx is not None:
                labels[start_idx:end_idx] = -100
        if "llama" in args.model_path.lower():
            instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mistral" in args.model_path.lower():
            instruction_template = r"[INST]"
            response_template = r"[/INST]"
        elif "qwen" in args.model_path.lower():
            instruction_template = r"<|im_start|>user\n"
            response_template = r"<|im_start|>assistant\n"
        user_tokens = tokenizer.encode(instruction_template, add_special_tokens=False)
        assistant_tokens = tokenizer.encode(response_template, add_special_tokens=False)
        for idx in range(len(input_ids)):
            if labels[idx:idx + len(user_tokens)].tolist() == user_tokens:
                labels[idx:idx + len(user_tokens)] = -100  # Mask user prompt
            # Check for assistant tokens
            elif labels[idx:idx + len(assistant_tokens)].tolist() == assistant_tokens:
                labels[idx:idx + len(assistant_tokens)] = -100

        res_labels[f"{prefix}lm_labels"] = labels.tolist()

    return res_labels  # Convert tensor to list for dataset







def format_labels(examples):
    # tokenizer = AutoTokenizer.from_pretrained(args)
    if "llama" in args.model_path.lower():
        tokenizer.pad_token = "<|end_of_text|>"
    elif "mistral" in args.model_path.lower():
        tokenizer.pad_token = tokenizer.unk_token

    tokenizer.padding_side = "right"
    ignore_index = -100
    # instruction_template = "<|im_start|>user"
    if "llama" in args.model_path.lower():
        instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
        # response_template = '<|im_start|>assistant'
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "mistral" in args.model_path.lower():
        instruction_template = r"[INST]"
        response_template = r"[/INST]"
    elif "qwen" in args.model_path.lower():
        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_start|>assistant\n"
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    instruction_token_ids = tokenizer.encode(instruction_template, add_special_tokens=False)

    batch = tokenizer(examples, padding=True, truncation=False, return_tensors="pt", add_special_tokens=False)
    labels = batch["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    for i in range(len(examples)):
        response_token_ids_idxs = []
        human_token_ids_idxs = []

        for assistant_idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                response_token_ids
                == batch["labels"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist()
            ):
                response_token_ids_idxs.append(assistant_idx + len(response_token_ids))

        if len(response_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{response_template}` in the "
                f'following instance: {tokenizer.decode(batch["input_ids"][i][-10:])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            batch["labels"][i, :] = ignore_index

        human_token_ids = instruction_token_ids
        for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
            # find the indexes of the start of a human answer.
            if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                human_token_ids_idxs.append(human_idx)

        if len(human_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find instruction key `{instruction_template}` in the "
                f'following instance: {tokenizer.decode(batch["input_ids"][i][:100])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            batch["labels"][i, :] = ignore_index

        if (
            len(human_token_ids_idxs) > 0
            and len(response_token_ids_idxs) > 0
            and human_token_ids_idxs[0] > response_token_ids_idxs[0]
        ):
            human_token_ids_idxs = [0] + human_token_ids_idxs

        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            # Make pytorch loss function ignore all non response tokens
            if idx != 0:
                batch["labels"][i, start:end] = ignore_index
            else:
                batch["labels"][i, :end] = ignore_index

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            batch["labels"][i, human_token_ids_idxs[-1] :] = ignore_index
    return batch


def build_chat_format(prompt, question, answer):
    user_prompt = prompt + "\n\n" + question if prompt else question
    return [
        {'role':"user", 'content': user_prompt},
        {'role':"assistant", 'content': answer}
    ]


def apply_chat_template(
    example, tokenizer, task="dpo"
):
    batch = {}
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task == "dpo":
        # print(example.keys())
        # print("--------------------")
        # print("using dpo")
        if all(k in example.keys() for k in ("chosens", "rejecteds")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            # prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            chosen_prompt_messages = []
            rejected_prompt_messages = []
            chosen_short_prompt_messages = []
            rejected_short_prompt_messages = []
            for i in range(len(example['questions'])):
                prompt = example['prompt'] if i == 0 else ""
                question = example['questions'][i]
                chosen_answer = example["chosens"][i]
                rejected_answer = example["rejecteds"][i]
                chosen_prompt_messages.extend(build_chat_format(prompt, question, chosen_answer))
                rejected_prompt_messages.extend(build_chat_format(prompt, question, rejected_answer))

                short_prompt = example['short_prompts'][i]
                # Note it is append here
                chosen_short_prompt_messages.append(build_chat_format(short_prompt, question, chosen_answer))
                rejected_short_prompt_messages.append(build_chat_format(short_prompt, question, rejected_answer))
            
            chosen_prompt_input = tokenizer.apply_chat_template(
                chosen_prompt_messages, tokenize=False, add_generation_prompt=False
            )
            rejected_prompt_input = tokenizer.apply_chat_template(
                rejected_prompt_messages, tokenize=False, add_generation_prompt=False
            )
            
            chosen_short_prompt_input = [tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=False
            ) for item in chosen_short_prompt_messages]
            rejected_short_prompt_input = [tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=False
            ) for item in rejected_short_prompt_messages]
            long_batch_chosen = format_labels([chosen_prompt_input])
            long_batch_rejected = format_labels([rejected_prompt_input])
            short_batch_chosen = format_labels(chosen_short_prompt_input)
            short_batch_rejected = format_labels(rejected_short_prompt_input)

            # short_batch = format_labels(chosen_short_prompt_input + rejected_short_prompt_input)
            for index in range(2):
                tmp_batch_chosen = long_batch_chosen if index == 0 else short_batch_chosen
                tmp_batch_rejected = long_batch_rejected if index == 0 else short_batch_rejected
                # len_chosen = 1 if index == 0 else len(chosen_short_prompt_messages)
                for k, toks in {
                    "chosen_": tmp_batch_chosen,
                    "rejected_": tmp_batch_rejected,
                }.items():
                    for type_key, tokens in toks.items():
                        tokens = tokens.tolist()
                        if type_key == "token_type_ids":
                            continue
                        if index == 1:
                            batch[f"ref_{k}{type_key}"] = tokens
                        else:
                            if len(tokens) != 1:
                                print(f"{k}{type_key}")
                            batch[f"{k}{type_key}"] = tokens[0] if len(tokens) == 1 else tokens
            return batch

            
            
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )


def load_json_data(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data.append(json.loads(item))
    return data





import re
from collections import Counter

def is_meaningless_response(response, repetition_threshold=0.95, min_word_length=3):
    # Normalize the response to lower case and remove punctuation
    response_clean = re.sub(r'[^\w\s]', '', response.lower())
    words = response_clean.split()

    if len(words) == 0:
        return True  # Empty response considered meaningless

    # Count total words and unique words
    total_words = len(words)
    word_counts = Counter(words)
    # print("Total words is: ", total_words)
    
    # Calculate the repetition ratio
    repeated_words = sum(count for word, count in word_counts.items() if count > 1)
    repetition_ratio = repeated_words / total_words
    
    # Check the average word length
    long_words = [word for word in words if len(word) >= min_word_length]
    
    # Determine if the response is considered "meaningless"
    is_repetitive = repetition_ratio > repetition_threshold
    is_short = len(long_words) < 2  # Less than 2 meaningful words
    
    return is_repetitive or is_short

def filter_censored_responses(data):
    """
    Filters out responses that contain any of the censored phrases.

    Parameters:
    responses (list of str): List of responses to filter.
    censored_phrases (list of str): List of phrases indicating a response is censored.

    Returns:
    list of str: List of non-censored responses.
    """
    rej_responses = [item['rejected'] for item in data]
    chosen_responses = [item['chosen'] for item in data]
    filtered_data = []
    for i, response in enumerate(rej_responses):
        if not response:
            continue
        if not chosen_responses[i]:
            continue
        if is_meaningless_response(response, repetition_threshold=0.95) or is_meaningless_response(chosen_responses[i], repetition_threshold=0.9):
            # print(response)
            # print("--------------------")
            # print(chosen_responses[i])
            continue
            # break
        # all_responses = response + chosen_responses[i]
        if not any(phrase.lower() in (response.lower() + "\n" + chosen_responses[i].lower()) for phrase in CENSORED_PHRASES):
            filtered_data.append(data[i])
    return filtered_data


import random
from collections import defaultdict
def sample_by_prompt_id(data_list, samples_per_group=2):
    # Convert dataset to list of dictionaries
    # data_list = dataset.to_list()  # Convert to Python dict for easy manipulation
    
    # Group by prompt_id
    groups = defaultdict(list)
    for item in data_list:
        # item = {key: data_list[key][idx] for key in data_list}
        groups[item['prompt_id']].append(item)
    
    # Sample 2 items from each group
    sampled_data = []
    for prompt_id, items in groups.items():
        # if len(items) < samples_per_group:
        #     continue
        if len(items) > samples_per_group:
            sampled_data.extend(random.sample(items, samples_per_group))
        else:
            sampled_data.extend(items)  # Include all if less than samples_per_group
    
    # Convert back to the dataset format if necessary (Assuming all columns are of the same length)
    # sampled_dataset = []
    # for key in dataset.features.keys():
    #     sampled_dataset[key] = [item[key] for item in sampled_data]

    return sampled_data


def group_by_prompt_id(data_list):
    # Convert dataset to list of dictionaries
    # data_list = dataset.to_list()  # Convert to Python dict for easy manipulation
    
    # Group by prompt_id
    groups = defaultdict(list)
    for item in data_list:
        # item = {key: data_list[key][idx] for key in data_list}
        groups[item['prompt_id']].append(item)
        assert item['prompt'] == groups[item['prompt_id']][0]['prompt']
    
#     # Sample 2 items from each group
#     sampled_data = []
#     for prompt_id, items in groups.items():
#         # if len(items) < samples_per_group:
#         #     continue
#         if len(items) > samples_per_group:
#             sampled_data.extend(random.sample(items, samples_per_group))
#         else:
#             sampled_data.extend(items)  # Include all if less than samples_per_group
    
    # Convert back to the dataset format if necessary (Assuming all columns are of the same length)
    # sampled_dataset = []
    # for key in dataset.features.keys():
    #     sampled_dataset[key] = [item[key] for item in sampled_data]
    group_data_list = []
    count = 0
    for prompt_id, items in groups.items():
        tmp_item = {}
        # if len(items) != 4:
        #     print(len(items))
        #     count += 1
        #     continue
        tmp_item['prompt'] = items[0]['prompt']
        tmp_item['short_prompts'] = [item['short_prompt'] for item in items]
        tmp_item['questions'] = [item['question'] for item in items]
        tmp_item['chosens'] = [item['chosen'] for item in items]
        tmp_item['rejecteds'] = [item['rejected'] for item in items]
        group_data_list.append(tmp_item)
    # print(count)
    return group_data_list



def filter_long_response(item):
    len_list = []
    if len(item["ref_chosen_input_ids"]) != len(item["ref_rejected_input_ids"]):
        return False
    for i in range(len(item["ref_chosen_input_ids"])):
        len_list.append(max(len(item["ref_chosen_input_ids"][i]), len(item["ref_rejected_input_ids"][i])))
    for l in len_list:
        if l > 32768:
            return False
    return True


import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Generate instruction')
    # parser.add_argument('--domain', type=str, default='Book', help='Domain of the text', required=True)
    parser.add_argument("--split", type=int, default=0)
    # parser.add_argument('--model_name', type=str, default='mistral', help='Model Name', required=True)
    parser.add_argument('--model_path', type=str, default='EleutherAI/gpt-neo-2.7B', help='Model Path', required=True)
    parser.add_argument('--data_path', type=str, default='data', help='Input file', required=True)
    parser.add_argument('--save_path', type=str, default='save_data', help='Save Folder', required=True)
    return parser.parse_args()



import sys


args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

data = load_json_data(os.path.join(args.data_path, f"split_{args.split}.json"))
# data = load_json_data(f"/mnt/workspace/gzchen/data_generation/qwen_longdpo_v3/LongDPO-128k-iter1/split_{split}.json")

# print(len(data), data[0].keys())

data = filter_censored_responses(data)
print(f"Number of data: {len(data)}")
data = group_by_prompt_id(data)
print(f"Number of data after filtering: {len(data)}")



train_dataset = Dataset.from_list(data)
column_names = list(train_dataset.features)
train_dataset = train_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
    num_proc=10,
    remove_columns=column_names,
    desc="Formatting comparisons with prompt template",
)

train_dataset = train_dataset.filter(lambda item: filter_long_response(item))

train_dataset = train_dataset.map(add_lm_labels, num_proc=10)

print(f"Number of data after all filtering: {len(train_dataset)}")
train_dataset.save_to_disk(os.path.join(args.save_path, f"split_{args.split}"), num_proc=10)
# print(len(train_dataset))