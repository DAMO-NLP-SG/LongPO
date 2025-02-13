import os
from datasets import load_from_disk, DatasetDict, Dataset

import os
from datasets import load_from_disk, Dataset, concatenate_datasets

def load_and_merge_datasets(folder_path):
    # Initialize a list to store datasets
    datasets = []
    
    # Loop through all subdirectories in the main folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for dirname in dirnames:
            dataset_path = os.path.join(dirpath, dirname)
            try:
                datasets.append(load_from_disk(dataset_path))
            except Exception as e:
                print(f"Failed to load dataset from {dataset_path}: {e}")
    print(len(datasets))
    # Merge all datasets into one
    if datasets:
        # merged_dataset = datasets[0]  # Start with the first dataset
        # for dataset in datasets[1:]:
        merged_dataset = concatenate_datasets(datasets)  # Concatenate each dataset
        return merged_dataset
    else:
        raise ValueError("No datasets found in the specified folder.")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Generate instruction')
    # parser.add_argument('--domain', type=str, default='Book', help='Domain of the text', required=True)
    # parser.add_argument('--model_name', type=str, default='mistral', help='Model Name', required=True)
    parser.add_argument('--data_path', type=str, default='data', help='Input file', required=True)
    parser.add_argument('--save_path', type=str, default='save_data', help='Save Folder', required=True)
    return parser.parse_args()




def save_dataset_to_disk(dataset, save_path):
    dataset.save_to_disk(save_path, num_proc=10)

if __name__ == "__main__":
    args = parse_args()
    folder_path = args.data_path  # Replace with your folder path
    save_path = args.save_path  # Replace with your save path

    # Load and merge datasets
    merged_dataset = load_and_merge_datasets(folder_path)
    
    # Save the merged dataset to disk
    save_dataset_to_disk(merged_dataset, save_path)

    print("Datasets loaded, merged, and saved successfully.")
