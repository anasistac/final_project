import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

# preprocessing function
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')  # avoid punkt issues when working in codespace in github
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))

    # remove stopwords and lowercase
    cleaned_tokens = [
        word.lower() for word in tokens
        if word.lower() not in stop_words
    ]

    return cleaned_tokens

# process all cleaned subtitle files in data_cleaned and save tokens
def process_cleaned_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)

        if os.path.isfile(input_path) and file.endswith('.txt'):
            # read
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # preprocess
            tokens = preprocess_text(text)

            # save tokens
            base_name = os.path.splitext(file)[0]
            output_file = os.path.join(output_folder, base_name + '_tokens.txt')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(' '.join(tokens))

            print(f"Tokenized: {file} → {base_name + '_tokens.txt'}")

# combine all tokenized files into one master file
def combine_tokenized_files(input_folder, output_path):
    all_tokens = []

    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)

        if os.path.isfile(file_path) and file.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                tokens = f.read().split()
                all_tokens.extend(tokens)

    # write combined tokens to one file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(all_tokens))

    print(f"Combined {len(all_tokens)} tokens into: {output_path}")

# run
process_cleaned_folder('data_cleaned/disney', 'data_preprocessed/disney')
combine_tokenized_files('data_preprocessed/disney', 'data_preprocessed/disney.txt')
