'''
The script splits text files in the specified folders into smaller parts based on word count. 
The purpose of this is to increase the number of documents for topic modeling, using LDA. 
'''

# Reference:
# ChatGPT: https://chatgpt.com/share/6827523a-2430-800e-8cc3-6583e0d0cf5c


# Imports
import os
from pathlib import Path
from math import ceil

def split_script_by_words(file_path, output_dir, parts=4):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    words = text.split()
    chunk_size = ceil(len(words) / parts)
    
    for i in range(parts):
        start = i * chunk_size
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        base_name = Path(file_path).stem
        output_file = os.path.join(output_dir, f"{base_name}_part{i+1}.txt")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(chunk_text)

def process_folder(input_folder, output_folder, parts=4):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            full_path = os.path.join(input_folder, filename)
            split_script_by_words(full_path, output_folder, parts)

# Original folders
disney_folder = "data_preprocessed/disney"
ghibli_folder = "data_preprocessed/ghibli"

# NEW target folders for split scripts
split_disney_folder = "data_split/disney"
split_ghibli_folder = "data_split/ghibli"

# Run the split
process_folder(disney_folder, split_disney_folder)
process_folder(ghibli_folder, split_ghibli_folder)
