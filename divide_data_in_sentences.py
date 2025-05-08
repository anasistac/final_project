import os
import nltk
import re

def simple_sentence_split(text):
    # splits sentences on period, exclamation, or question mark followed by space and a capital letter
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())


def split_subtitles_into_sentences(input_folder, output_folder, combined_output_file):
    os.makedirs(output_folder, exist_ok=True)
    all_sentences = []

    for filename in os.listdir(input_folder):
        if os.path.isfile(os.path.join(input_folder, filename)):
            input_path = os.path.join(input_folder, filename)

            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # join all lines into one string
            text = ' '.join(line.strip() for line in lines)
            sentences = simple_sentence_split(text)


            # split into sentences
            sentences = simple_sentence_split(text)
            all_sentences.extend(sentences)

            # save sentences for current movie
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, base_name + '_sentences.txt')
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for sentence in sentences:
                    f_out.write(sentence.strip() + '\n')

            print(f"Split {filename} into {len(sentences)} sentences → {output_path}")

    # save all combined sentences into one file
    with open(combined_output_file, 'w', encoding='utf-8') as f_combined:
        for sentence in all_sentences:
            f_combined.write(sentence.strip() + '\n')

split_subtitles_into_sentences(
    input_folder='data_cleaned/disney',
    output_folder='data_in_sentences/disney',
    combined_output_file='data_in_sentences/disney_sentences.txt'
)
