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
Based on the content presented above, generate 4 comprehensive English questions that test a reader's comprehension, analytical skills, and ability to extract and interconnect key themes and ideas across the entire document.

Each question should:
1. Encourage the reader to draw connections between different sections or concepts within the text.
2. Challenge the reader to not only recall information but also to synthesize and summarize the material in a coherent manner.
3. Be unique in its focus, avoiding repetition and ensuring a broad coverage of the document's content.
4. Stimulate critical thinking by requiring the application or evaluation of the text's information in broader contexts or hypothetical scenarios, if relevant.
5. Ensure the question is clear and unambiguous.

Please directly give the questions without verbose illustration, and format the 4 questions numerically from “1:” to “4:”.
"""


CHAR_RATIO = 3.6
CHUNK_LEN =16384
MIN_DOC_LEN = 128*1024
NUM_QUESTION = 4

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



def parse_args():
    parser = argparse.ArgumentParser(description='Generate instruction')
    parser.add_argument('--model_name', type=str, default='vllm', help='Model name')
    parser.add_argument('--model_path', type=str, default='models/vllm', help='Model path')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top k sampling')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top p sampling')
    parser.add_argument('--max_length', type=int, default=1024, help='Max length of the instruction')
    parser.add_argument('--max_chunk_length', type=int, default=32768, help='Max length of the chunk data')
    parser.add_argument('--min_doc_length', type=int, default=65536, help='Min length of the document')
    parser.add_argument('--process_length', type=int, default=131072, help='The process length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # parser.add_argument('--prompt', type=str, default='A', help='Prompt')
    parser.add_argument('--output', type=str, default='data_generation', help='Output file')
    parser.add_argument('--domain', type=str, default='Book', help='Domain of the text')
    parser.add_argument('--data_path', type=str, default='data', help='Input file')
    parser.add_argument('--save_path', type=str, default='save_data', help='Save Folder')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='gpus')
    parser.add_argument('--world_size', type=int, default=8, help='gpus')
    parser.add_argument('--rank', type=int, default=0, help='gpus')
    return parser.parse_args()






def sample_data_to_target_count(chunk_data, target_count):
    """
    Randomly sample lists from a list of lists to get as close to the target item count as possible without going over.

    :param lists_of_items: A list of lists containing items.
    :param target_item_count: The target total number of items desired.
    :return: A list of randomly sampled lists.
    """

    # First, shuffle the list to ensure randomness since we will be checking in order
    random.shuffle(chunk_data)

    sampled_lists = []
    current_count = 0

    for chunk in chunk_data:
        # Only add the list if it doesn't exceed the target count
        if current_count <= target_count:
            sampled_lists.append(chunk)
            current_count += len(chunk)

    return sampled_lists


def load_json_file_naive(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data_item = json.loads(item)
            # if len(data_item['text']) > MIN_DOC_LEN*CHAR_RATIO and len(data_item['text']) < MIN_DOC_LEN*CHAR_RATIO:
            data.append(data_item)
    # if len(data) > 40000:
    #     data = random.sample(data, 40000)
    return data


def load_json_file(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data_item = json.loads(item)
            # if len(data_item['text']) > MIN_DOC_LEN*CHAR_RATIO and len(data_item['text']) < MIN_DOC_LEN*CHAR_RATIO:
            data.append({'text': data_item})
    # if len(data) > 40000:
    #     data = random.sample(data, 40000)
    return data





def split_paragraphs(text):
    # Split the text by two or more newlines which often indicate paragraph breaks
    paragraphs = re.split(r'\n\n+', text)
    # Remove any leading/trailing whitespace
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

def split_sections(text):
    section_pattern = re.compile(r'\\(sub)*section\{([^}]*)\}', re.IGNORECASE)

    # Find all matches of the section pattern
    matches = list(re.finditer(section_pattern, text))
    sections = []
    for i, match in enumerate(matches):
        # Extract the current section or subsection title
        title = match.group(0).strip()  # This includes the full LaTeX command like \section{...}
        # Find the start of the current section
        start = match.start()
        # If this is not the last match, the end is the start of the next match, otherwise it's the end of the document
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Print the section or subsection title and its content
        
        sections.append(text[start:end].strip())
    return sections

def group_paragraphs_by_tokens(paragraphs, tokenizer, max_tokens, min_tokens):
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = tokenizer(paragraph, truncation=False)["input_ids"]
        paragraph_length = len(paragraph_tokens)
        
        if current_chunk_length + paragraph_length <= max_tokens:
            current_chunk.append(paragraph)
            current_chunk_length += paragraph_length
        else:
            if current_chunk_length >= min_tokens and current_chunk_length <= max_tokens:
                chunks.append(' '.join(current_chunk))
            current_chunk = [paragraph]
            current_chunk_length = paragraph_length

    if current_chunk_length >= min_tokens and current_chunk_length <= max_tokens:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_book_item(book_item, tokenizer, max_tokens, min_tokens):
    paragraphs = split_paragraphs(book_item['text'])
    return group_paragraphs_by_tokens(paragraphs, tokenizer, max_tokens, min_tokens)

def process_arxiv_item(arxiv_item, tokenizer, max_tokens, min_tokens):
    paragraphs = split_sections(arxiv_item['text'])
    return group_paragraphs_by_tokens(paragraphs, tokenizer, max_tokens, min_tokens)

def load_data(data_path, domain, rank, world_size, tokenizer, max_tokens, min_tokens):
    json_data = load_json_file(os.path.join(data_path, f"{domain.lower()}/split_128k_filter.json"))
    json_data= [json_data[i::world_size] for i in range(world_size)][rank]
    if domain == "Book" or domain == "Code":
        with Pool(20) as pool:
            chunk_data = pool.starmap(process_book_item, [(item, tokenizer, max_tokens, min_tokens) for item in json_data])
    elif domain == "Arxiv":
        with Pool(20) as pool:
            chunk_data = pool.starmap(process_arxiv_item, [(item, tokenizer, max_tokens, min_tokens) for item in json_data])
    return chunk_data




# def extract_questions(generation_text):
#     # Define a regex pattern to match lines that look like question headers
#     # This pattern matches line starts, followed by one or more digits,
#     # followed by any special characters and optional whitespace.
#     # It captures until the next occurrence of a similar pattern.
#     pattern = re.compile(r'\b\d+\s*[:.!?-]+\s*(.+?)\s*(?=\b\d+\s*[:.!?-]+\s*|$)', re.DOTALL)
    
#     # Find all matches of the pattern and extract them
#     questions = pattern.findall(generation_text)
    
#     # Filter out empty strings and trim whitespace from each question
#     questions = [question.strip() for question in questions if question.strip()]
    
#     return questions

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

def flatten_chunks(chunk_data):
    flatted_data = []
    index_mapping = []
    for i, chunks in enumerate(chunk_data):
        for j, item in enumerate(chunks):
            flatted_data.append({'text': item})
            index_mapping.append(
                {
                "chunk_id": i,
                "segment_id": j
                }
            )
    return flatted_data, index_mapping






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

    question_sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=args.max_length)

    answer_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_length)
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
        gpu_memory_utilization=0.95
    )
    # llm.llm_engine.tokenizer.eos_token_id = 1
    def build_instruction_generation_prompts(example):
        input_prompt =  "Context:\n" + example['text'] + "\n\n" + QUERT_PROMPT
        # input_prompt =  example['prompt'] + "\n\n" + example['question']
        chat = tokenizer.apply_chat_template([{"role": "user", "content": input_prompt}], tokenize=False, add_generation_prompt=True)
        return {"input_prompt": chat}



    # CHUNK_LEN = args.chunk_length
    chunk_data = load_data(args.data_path, args.domain, args.rank, args.world_size, tokenizer, max_tokens=30000, min_tokens=16384)
    print(f"Load {args.domain} data sucessfully! Number of chunks: {len(chunk_data)}. Number of segments: {sum([len(item) for item in chunk_data])}")

    all_chunks, index_mapping = flatten_chunks(chunk_data)


    question_dataset = Dataset.from_list(all_chunks)



    question_dataset = question_dataset.map(
        build_instruction_generation_prompts,
        num_proc=20
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
        num_proc=10,
    )["input_ids"]

    self_generated_questions = llm.generate(sampling_params=question_sampling_params, prompt_token_ids=question_input_ids)

    extract_questions_list = []
    # Print the outputs.
    filter_num = 0
    for i, output in enumerate(self_generated_questions):
        generated_text = output.outputs[0].text
        if len(question_input_ids[i]) > args.max_chunk_length:
            filter_num += 1
            continue
        extract_questions_list.append(
            {
            "generated_questions": generated_text,
            "chunk_id": index_mapping[i]['chunk_id'],
            "segment_id": index_mapping[i]['segment_id']
            }
        )


    print(f"Generate {args.domain} questions sucessfully! Number of questions: {len(extract_questions_list)}! Filter num: {filter_num}.")
    save_jsonl(extract_questions_list, f"{args.save_path}/{args.domain}/{args.process_length/1024}k_chunks/split_{args.rank}/{args.domain}_question.json")
    save_jsonl(chunk_data, f"{args.save_path}/{args.domain}/{args.process_length/1024}k_chunks/split_{args.rank}/{args.domain}_chunk_data.json")
    # extract_questions_list = load_json_file_naive(f"data_generation/mistral_longdpo_v2/{args.domain}/32k_chunks/split_{args.rank}/{args.domain}_question.json")
    
    # print(f"Save {args.domain} questions and chunk data sucessfully!")

    generated_questions = [extract_questions(item['generated_questions']) for item in extract_questions_list]
    # question_prompts, question_index_mapping = flatten_chunks(generated_questions)
    question_index_mapping = []
    question_prompts = []
    for i, questions in enumerate(generated_questions):
        # question_index_mapping.extend([i]*len(questions))
        if len(questions) != NUM_QUESTION:
            continue
        question = random.choice(questions)
        # for question in questions:
        question_index_mapping.append(i)
        phase = random.choice(PHRASES)
        question_prompt = all_chunks[i]['text'] + f"\n\n{phase} {question}"
        question_prompts.append(
            {
            "text": question_prompt,
            "question": question
            }
        )
    def build_answer_generation_prompt(example):
        input_prompt =  example['text']
        # input_prompt =  example['prompt'] + "\n\n" + example['question']
        chat = tokenizer.apply_chat_template([{"role": "user", "content": input_prompt}], tokenize=False, add_generation_prompt=True)
        return {"input_prompt": chat}
    
    answer_dataset = Dataset.from_list(question_prompts)
    answer_dataset = answer_dataset.map(
        build_answer_generation_prompt,
        num_proc=10
    )
    answer_input_ids = answer_dataset.map(
        tokenize_func,
        batched=True,
        batch_size=100,
        num_proc=10,
    )["input_ids"]

    answers = llm.generate(sampling_params=answer_sampling_params, prompt_token_ids=answer_input_ids)
    answer_list = []
    for i, output in enumerate(answers):
        # prompt = output.prompt
        generated_text = output.outputs[0].text
        answer_list.append(
            {
            # "prompt": prompt, 
            "ori_question": question_prompts[i]['question'],
            "question": question_prompts[i]['text'],
            "generated_answer": generated_text,
            "chunk_id": index_mapping[question_index_mapping[i]]['chunk_id'],
            "segment_id": index_mapping[question_index_mapping[i]]['segment_id']
            }
        )
    # with open(f"{args.output}/{args.domain}_qa.json", 'w') as f:
    #     for item in answer_list:
    #         f.write(json.dumps(item) + "\n")
    save_jsonl(answer_list, f"{args.save_path}/{args.domain}/{args.process_length/1024}k_chunks/split_{args.rank}/{args.domain}_qa.json")
    # for question
    print(f"Generate {args.domain} questions and answers sucessfully! Number of answers: {len(answer_list)}")

    
    
    



if __name__ == "__main__":
    main()