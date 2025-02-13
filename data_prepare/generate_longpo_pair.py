from vllm import LLM, SamplingParams
import argparse

import os
import re

import json
from multiprocessing import Pool
import random
from transformers import AutoTokenizer, set_seed
from datasets import Dataset

QUERT_PROMPT ="""
Based on the content presented above, generate 5 comprehensive English questions that test a reader's comprehension, analytical skills, and ability to extract and interconnect key themes and ideas across the entire document.

Each question should:
1. Encourage the reader to draw connections between different sections or concepts within the text.
2. Challenge the reader to not only recall information but also to synthesize and summarize the material in a coherent manner.
3. Be unique in its focus, avoiding repetition and ensuring a broad coverage of the document's content.
4. Stimulate critical thinking by requiring the application or evaluation of the text's information in broader contexts or hypothetical scenarios, if relevant.

Please format the 5 questions numerically from “1:” to “5:”, and ensure they are sufficiently open-ended to allow for in-depth responses.
"""

CHAR_RATIO = 3.4
CHUNK_LEN =80000
# MIN_DOC_LEN = 120000

PHRASES = [
    "As described in the preceding text,",
    "As detailed in the foregoing document,",
    "As mentioned earlier in the text,",
    "As specified in the above material,",
    "As indicated in the preceding pages,",
    "In accordance with the above content,",
    "As set forth in the document mentioned,",
    "Following the details provided above,",
    "As elaborated in the prior document,",
    "Pursuant to the information above,",
    "In line with the preceding information,",
    "Reflecting on the content above,",
    "Based on the preceding descriptions,",
    "With reference to the document above,",
    "Drawing from the details above,",
    "Considering the information outlined above,",
    "As articulated in the previous sections,",
    "Conforming to the descriptions above,",
    "As captured in the above explanations,",
    "As chronicled in the earlier document,"
]


import random
from collections import defaultdict
def sample_by_prompt_id(data_list, samples_per_group=8):
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


def parse_args():
    parser = argparse.ArgumentParser(description='Generate instruction')
    parser.add_argument('--model_name', type=str, default='vllm', help='Model name')
    parser.add_argument('--model_path', type=str, default='models/vllm', help='Model path')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top k sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top p sampling')
    parser.add_argument('--max_length', type=int, default=1024, help='Max length of the instruction')
    parser.add_argument('--chunk_length', type=int, default=7000, help='Max length of the chunk data')
    parser.add_argument('--min_doc_length', type=int, default=120000, help='Min length of the document')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--process_length', type=int, default=131072, help='The process length')
    parser.add_argument('--output', type=str, default='data_generation', help='Output file')
    parser.add_argument('--domain', type=str, default='Book', help='Domain of the text')
    parser.add_argument('--data_path', type=str, default='data', help='Input file')
    parser.add_argument('--save_path', type=str, default='data_generation', help='Output file')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='gpus')
    parser.add_argument('--world_size', type=int, default=8, help='gpus')
    parser.add_argument('--rank', type=int, default=0, help='gpus')
    return parser.parse_args()






def load_json_file(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data_item = json.loads(item)
            data.append(data_item)
    return data




def extract_questions(generation_text):
    # Define a regex pattern to match lines that look like question headers
    # This pattern matches line starts, followed by one or more digits,
    # followed by any special characters and optional whitespace.
    # It captures until the next occurrence of a similar pattern.
    pattern = re.compile(r'\b\d+\s*[:.!?-]+\s*(.+?)\s*(?=\b\d+\s*[:.!?-]+\s*|$)', re.DOTALL)
    
    # Find all matches of the pattern and extract them
    questions = pattern.findall(generation_text)
    
    # Filter out empty strings and trim whitespace from each question
    questions = [re.sub(r'^[\*\s#.!]*(.+?)[\*\s#.!]*$', r'\1', question.strip(), flags=re.DOTALL) for question in questions if question.strip()]
    
    return questions



def save_jsonl(data, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")





def main():
    args = parse_args()
    set_seed(args.seed)


    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        use_fast=True,
        truncation=True,
        model_max_length=args.process_length,
        truncation_side="right"
    )
    llm = LLM(
        model=args.model_path, 
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size, 
        seed=args.seed, 
        dtype="bfloat16",
        gpu_memory_utilization=0.9
    )
    # llm.llm_engine.tokenizer.eos_token_id = 1
    def build_instruction_generation_prompts(example):
        # input_prompt =  "\Content:\n" + example['text'] + "\n\n" + QUERT_PROMPT
        input_prompt =  example['prompt'] + "\n\n" + example['question']
        chat = tokenizer.apply_chat_template([{"role": "user", "content": input_prompt}], tokenize=False, add_generation_prompt=True)
        return {"input_prompt": chat}


    data_path = args.data_path
    sft_data_all = load_json_file(data_path)
    world_size = args.world_size
    sft_data = [sft_data_all[i::world_size] for i in range(world_size)][args.rank]

    prompt_list = []

    for i, item in enumerate(sft_data):
        prompt = sft_data[i]['prompt']
        for j in range(0, len(item["messages"]), 2):
            assert item["messages"][j]["role"] == "user"
            assert item["messages"][j+1]["role"] == "assistant"
            prompt_item = {
                "short_prompt": item["messages"][j]['chunk_prompt'],
                "prompt": prompt,
                "question": item["messages"][j]["content"] ,
                # "chunk_id":,
                "prompt_id": i * world_size + args.rank,
                # "segment_id":,
                "chosen": item["messages"][j+1]["content"],
                "rejected": ""
            }
            prompt_list.append(prompt_item)
    prompt_list = sample_by_prompt_id(prompt_list, 2)
    question_dataset = Dataset.from_list(prompt_list)



    question_dataset = question_dataset.map(
        build_instruction_generation_prompts,
        num_proc=40
    )

    def tokenize_func(examples):
        return tokenizer(examples['input_prompt'], truncation=True)

    """
    Generate questions
    """


    question_input_ids = question_dataset.map(
        tokenize_func,
        batched=True,
        batch_size=100,
        num_proc=40,
    )["input_ids"]
    print(len(question_input_ids[0]), len(question_input_ids[10]))
    self_generated_questions = llm.generate(sampling_params=sampling_params, prompt_token_ids=question_input_ids)

    extract_questions_list = []
    # Print the outputs.
    for i, output in enumerate(self_generated_questions):
        generated_text = output.outputs[0].text
        prompt_list[i]["rejected"] = generated_text

        
    # print(f"Generate {args.domain} questions sucessfully! Number of questions: {len(extract_questions_list)}")
    
    save_jsonl(prompt_list, f"{args.save_path}/split_{args.rank}.json")

    



if __name__ == "__main__":
    main()