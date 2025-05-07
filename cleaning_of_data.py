import os
import re

def clean_srt_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    cleaned_lines = []

    for line in lines:

        # remove invisible BOM or Unicode markers (1 in the beginning)
        line = re.sub(r'[\u200b\uFEFF\u202a\u202c\u202d\u200e\u200f]', '', line)

        line = line.strip()

        # skip block numbers
        if re.match(r'^\d+$', line):
            continue

        # skip timestamps -> all the same format: (00:00:00,000 --> 00:00:00,000)
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
            continue

        # remove dashes at beginning of sentence
        line = re.sub(r'^\s*-+', '', line).strip()


        # skip empty lines
        if line == '':
            continue

        cleaned_lines.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

# Example use:
clean_srt_file(
    input_path='data2/disney/Dumbo',
    output_path='data_cleaned/disney/Dumbo_cleaned.txt'
)
