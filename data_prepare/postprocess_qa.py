import os
import json
import re
import argparse
from multiprocessing import Pool
import random
from datasets import Dataset

def save_jsonl(data, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
def load_json_data(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data.append(json.loads(item))
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




def extract_questions_data(qa_data, question_data):
    generated_questions = [extract_questions(item['generated_questions']) for item in question_data]
    all_questions = []
    for i, questions in enumerate(generated_questions):
        # question_index_mapping.extend([i]*len(questions))
        if len(questions) != 4:
            continue
        for question in questions:
            all_questions.append(question)
    new_qa_data = []
    for i, item in enumerate(qa_data):
        assert item["question"].endswith(all_questions[i])
        if i < 2:
            print(item["question"].endswith(all_questions[i]))
        item["ori_question"] = all_questions[i]
        new_qa_data.append(item)
    return new_qa_data

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


def check_prompt(chunk_data, new_qa_data):
    for item in new_qa_data:
        assert item['question'].startswith(chunk_data[item['chunk_id']][item['segment_id']])


def preprocess_json_file(folder_path, domain, start_idx=0):
    """
    Merges all JSON files in the specified folder into a single JSON file.

    Parameters:
    - folder_path: string, path to the folder containing the JSON files
    - output_file: string, path to the output JSON file

    Returns:
    - None
    """
    merged_data = []
    
    chunk_data = load_json_data(os.path.join(folder_path, f"{domain}_chunk_data.json"))
    qa_data = load_json_data(os.path.join(folder_path, f"{domain}_qa.json"))
    question_data = load_json_data(os.path.join(folder_path, f"{domain}_question.json"))
    
    # new_qa_data = extract_questions_data(qa_data, question_data)
    check_prompt(chunk_data, qa_data)
    for item in qa_data:
        item['chunk_id'] += start_idx
    # save_path = os.path.join(f"/mnt/workspace/Projects/LongAlign/data_generation/mistral_longdpo_v2/{domain}/32k_chunks/", f"{domain}_qa.json")
    # save_jsonl(new_qa_data, save_path)
    return chunk_data, qa_data

    

def merge_json_files(folder_path, domain):
    # folder_path = 
    chunk_data = []
    qa_data = []
    start_idx = 0
    for filename in os.listdir(folder_path):
        if filename.startswith("split_"):
            file_path = os.path.join(folder_path, filename)
            split_chunk_data, split_qa_data = preprocess_json_file(file_path, domain, start_idx)
            chunk_data.extend(split_chunk_data)
            qa_data.extend(split_qa_data)
            start_idx += len(split_chunk_data)
    check_prompt(chunk_data, qa_data)
    qa_save_path = os.path.join(folder_path, f"{domain}_qa.json")
    save_jsonl(qa_data, qa_save_path)
    chunk_save_path = os.path.join(folder_path, f"{domain}_chunk_data.json")
    save_jsonl(chunk_data, chunk_save_path)
    return qa_data, chunk_data

# domain = "Book" 
# merge_json_files(f"{domain}/128k_chunks/", domain)
    # Iterate through all files in the specified folder
#     for filename in os.listdir(folder_path):
#         if filename.beginwith("split_"):
#             file_path = os.path.join(folder_path, filename)
            
#             # Read each JSON file
#             # with open(file_path, 'r') as f:
#             try:
#                 data = load_json_data(file_path)
#                 if isinstance(data, list):
#                     merged_data.extend(data)
#                 else:
#                     merged_data.append(data)
#             except json.JSONDecodeError as e:
#                 print(f"Error reading {filename}: {e}")
#     save_path = os.path.join("/mnt/workspace/Projects/LongAlign/data_generation/mistral_longdpo_v2/Arxiv/book/", output_file)
#     save_jsonl(merged_data, save_path)
    # Write the merged data to the output file
    # with open(output_file, 'w') as f:
    #     json.dump(merged_data, f, indent=4)
    
    # print(f"Merged {len(merged_data)} items into {output_file}")

# Example usage:
# domain = "General"
# folder_path = "32k_chunks"
# # for folder_path in ["128k", "512k", "256k", "1M"]:
#     # folder_path = '128k'
# output_file = f'{folder_path}.json'
# merge_json_files(folder_path, output_file)



import random
import json
import os
from datasets import Dataset

def load_json_data(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data.append(json.loads(item))
    return data

def save_jsonl(data, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


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

import re
from collections import Counter

def is_meaningless_response(response, repetition_threshold=0.5, min_word_length=3):
    # Normalize the response to lower case and remove punctuation
    response_clean = re.sub(r'[^\w\s]', '', response.lower())
    words = response_clean.split()

    if len(words) == 0:
        return True  # Empty response considered meaningless

    # Count total words and unique words
    total_words = len(words)
    word_counts = Counter(words)
    
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
    responses = [item['rejected'] for item in data]
    filtered_data = []
    for i, response in enumerate(responses):
        if not response:
            continue
        if is_meaningless_response(response):
            continue
        if not any(phrase.lower() in response.lower() for phrase in CENSORED_PHRASES):
            filtered_data.append(data[i])
    return filtered_data


def group_data_for_sft(answer_data, chunk_data):
    # question_path = f"/mnt/workspace/Projects/LongAlign/data_generation/zephyr-gemma/{domain}/generated_questions.json"
    # answer_path = f"{folder}/{domain}/128k_chunks/{domain}_qa.json"
    # chunk_data_path = f"{folder}/{domain}/128k_chunks/{domain}_chunk_data.json"
    # answer_data = load_json_data(answer_path)
    # chunk_data = load_json_data(chunk_data_path)
    chat_data = []
    for i, item in enumerate(chunk_data):
        prompt = "\n\n\n".join(item)
        chat_data.append(
            {
                "chunk_id": i,
                "prompt": prompt,
                "messages": []
            }
        )
    
    # answer_data = filter_long_data(answer_data)
    for answer in answer_data:
        chunk_id = answer['chunk_id']
        prompt = chat_data[chunk_id]["prompt"]
        # phrase = random.choice(PHRASES)
        # assert item['question'].startswith(chunk_data[item['chunk_id']][item['segment_id']])
        # messages = chat_data[chunk_id]["messages"]
        if not answer['question'][:100] in prompt:
            # print(answer['question'][:10])
            continue
        question_item = answer['question'].split("\n\n")[-1]
        assert chunk_data[chunk_id][answer['segment_id']] in answer['question']
        assert chunk_data[chunk_id][answer['segment_id']] in chat_data[chunk_id]['prompt']
        if not any(question_item.startswith(pharse) for pharse in PHRASES):
            # print(answer['question'].split("\n\n")[-2:])
            continue
            # assert False
        pharse_index = -1
        for i, pharse in enumerate(PHRASES):
            if question_item.startswith(pharse):
                pharse_index = i
        chat_data[chunk_id]["messages"].append(
            {
                "role": "user",
                "chunk_prompt": chunk_data[chunk_id][answer['segment_id']],
                "content": PHRASES[pharse_index] + " " + answer['ori_question']
            }
        )
        chat_data[chunk_id]["messages"].append(
            {
                "role": "assistant",
                "content": answer["generated_answer"]
            }
        )
    
        # chat_data[chunk_id]["messages"] = messages
    chat_data = [item for item in chat_data if len(item["messages"]) >0 and len(item["messages"])%2==0]
    return chat_data
    
# group_data_for_sft("Book")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate instruction')
    parser.add_argument('--domain', type=str, default='Book', help='Domain of the text', required=True)
    parser.add_argument('--data_path', type=str, default='data', help='Input file', required=True)
    parser.add_argument('--save_path', type=str, default='save_data', help='Save Folder', required=True)
    return parser.parse_args()


import os


if __name__ == "__main__":
    args = parse_args()
    # src_folder_path = f"{args.process_length // 1024}k_chunks"
    src_folder_path = args.data_path
    res_save_path = args.save_path
    # output_file = f'{folder_path}.json'
    if not os.path.exists(f"{src_folder_path}/{args.domain}_chunk_data.json"):
        qa_data, chunk_data = merge_json_files(src_folder_path, args.domain)
    else:
        qa_data = load_json_data(f"{src_folder_path}/{args.domain}_qa.json")
        chunk_data = load_json_data(f"{src_folder_path}/{args.domain}_chunk_data.json")
    # qa_data, chunk_data = merge_json_files(src_folder_path, args.domain)
    chat_data = group_data_for_sft(qa_data, chunk_data)
    # filter_chat_data = filter_censored_responses(chat_data)
    print(len(chat_data))
    save_jsonl(chat_data, res_save_path)
