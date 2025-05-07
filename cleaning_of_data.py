import os
import re

def clean_srt_line(line):
    # remove invisible characters
    line = re.sub(r'[\u200b\uFEFF\u202a\u202c\u202d\u200e\u200f]', '', line).strip()

    # remove dashes at beginning of sentence
    line = re.sub(r'^\s*-+', '', line).strip()

    # remove HTML tags like <i> and </i>
    line = re.sub(r'</?i>', '', line)

    # remove content in brackes and parenthesis
    line = re.sub(r'\[.*?\]|\(.*?\)', '', line)

    # remove speaker names in all caps followed by a colon (example: "MIGUEL:")
    line = re.sub(r'^[A-Z\s]+:', '', line).strip()

    # remove ALL CAPS words (3 or more letters)
    line = re.sub(r'\b[A-Z]{3,}\b', '', line).strip()

    # skip block numbers
    if re.match(r'^\d+$', line):
        return None
    
    # skip timestamp lines
    if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
        return None
    
    # skip empty lines
    if line == '':
        return None

    return line

# function to process all SRT files in a folder and save cleaned versions
def clean_all_srt_files(input_dir, output_dir):

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # loop through all files of input directory
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            # create input and output paths
            input_path = os.path.join(input_dir, file)
            base_name = os.path.splitext(file)[0]  # removes .srt if present
            output_filename = base_name + '_cleaned.txt'
            output_path = os.path.join(output_dir, output_filename)

            # read all lines from subtitle file
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            cleaned_lines = []

            # clean each line using cleaning function
            for line in lines:
                cleaned = clean_srt_line(line)
                if cleaned:
                    cleaned_lines.append(cleaned)

            # write cleaned lines to new file
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in cleaned_lines:
                    f.write(line + '\n')
            
            # will delete later
            print(f"Cleaned: {file} → {output_filename}")

clean_all_srt_files(input_dir='data2/disney', output_dir='data_cleaned/disney')
