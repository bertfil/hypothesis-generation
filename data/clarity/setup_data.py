import os
import json
import random
from datasets import load_dataset

# Configuration
TASK_NAME = "clarity"
DATA_DIR = ""
SEED = 42

print(f"Downloading QEvasion dataset for task: {TASK_NAME}...")
# Load dataset from Hugging Face
dataset = load_dataset("ailsntua/QEvasion")

# The dataset only has 'train' and 'test'. We need to create 'val' from 'train'.
print("Splitting train set into train/val...")
train_full = dataset['train']
# Split: 85% Train, 15% Validation
train_test_split = train_full.train_test_split(test_size=0.15, seed=SEED)
train_data = train_test_split['train']
val_data = train_test_split['test']
test_data = dataset['test']

def convert_to_repo_format(ds):
    output = {
        "question": [],  # The Question
        "answer": [],  # The Answer
        "label": []   # The Clarity label
    }
    
    for row in ds:
        # Map the specific fields from QEvasion
        output["question"].append(row['interview_question'])
        output["answer"].append(row['interview_answer'])
        
        # 'classes' column contains the label (e.g., 'Evasive', 'Direct')
        # We ensure it's a string as the repo expects
        output["label"].append(str(row['clarity_label']))
        
    return output

# Process and save all splits
splits = {
    'train': train_data,
    'val': val_data,
    'test': test_data
}

for split_name, ds_split in splits.items():
    print(f"Processing {split_name} data ({len(ds_split)} examples)...")
    formatted_data = convert_to_repo_format(ds_split)
    
    output_path = os.path.join(DATA_DIR, f"{split_name}.json")
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    print(f"Saved to {output_path}")

print("\nSuccess! Data setup complete.")
print(f"Location: {os.path.abspath(DATA_DIR)}")
