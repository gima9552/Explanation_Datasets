# -*- coding: utf-8 -*-
"""Explanations_without_training.ipynb

Script for Classifying English Sentences WITHOUT example training

This script is designed to perform classification of English sentences
using pre-trained language models (Llama 3, 8billions parameter) via the Unslooth library. The classification
categories include Causation, Mechanistic Causation, Contrastive Explanation, Functional,
Correlation, Pragmatic Approach, and No Explanation. The script uses different knowledge
templates for classification, processes the test sentences, and outputs the classified results
to TSV files.

Description:
1. Reading the Dataset: Reads the English test set from a text file and loads it into a list of sentences.
2. Loading the Model: Loads a pre-trained language model and tokenizer from the Unslooth library.
3. Classifying Sentences: Classifies sentences based on specified guidelines stored in a template and stores the results.
4. Writing Results: Writes the original and classified sentences to new TSV files for each template.

Output:
The script generates TSV files named Classifications-{template_name}.tsv containing the original and classified sentences.

Dependencies:
- unsloth
- pandas
- torch
- datasets
"""
import os
import csv
from datasets import Dataset
import pandas as pd
from unsloth import FastLanguageModel

# List all files and directories in the current directory
base_path = './'
print(os.listdir(base_path))

# Define the path to the directory containing the dataset
directory_path = './Llama3-8b'

# Verify by listing contents
print(os.listdir(directory_path))

# Paths to the prompt template files in the current directory
template_paths = {
    'no_knowledge': os.path.join(directory_path, 'template_no_knowledge.txt'),
    'light_knowledge': os.path.join(directory_path, 'template_light_knowledge.txt'),
    'proportional_knowledge': os.path.join(directory_path, 'template_proportional_knowledge.txt')
}

# Function to load a template
def load_template(template_path):
    with open(template_path, 'r') as file:
        template = file.read()
    return template

# Load all templates and display them one by one
templates = {}
for name, path in template_paths.items():
    template = load_template(path)
    templates[name] = template
    print(f"Loaded template for {name}:")
    print(f"""{template}""")
    print("\n" + "="*50 + "\n")

# Path to the test set file
test_set_path = os.path.join(directory_path, 'test-300.txt')

# Load the test set
with open(test_set_path, 'r') as file:
    test_sentences = file.readlines()

# Strip newlines and any leading/trailing whitespace from each sentence
test_sentences = [sentence.strip() for sentence in test_sentences]

# Load the model and tokenizer
from unsloth import FastLanguageModel
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

EOS_TOKEN = tokenizer.eos_token

import re
import torch
from collections import Counter

# Set the random seed for reproducibility
torch.manual_seed(42)

# Function to classify a single sentence with the given prompt template and extract classifications
def classify_sentence_with_prompt(sentence, prompt_template, temperature=0.7, num_return_sequences=5):
    try:
        prompt_formatted = prompt_template.format(text=sentence)
    except KeyError as e:
        print(f"Template formatting error: {e}")
        print(f"Template: {prompt_template}")
        return "Template Error"
    except IndexError as e:
        print(f"Template formatting error: {e}")
        print(f"Template: {prompt_template}")
        return "Template Error"

    inputs = tokenizer(prompt_formatted, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        num_return_sequences=num_return_sequences
    )
    classified_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    all_classifications = []
    for classified_output in classified_outputs:
        match = re.search(r"### Response:\s*(.*)", classified_output, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
            classifications = re.findall(r"\b(Causation|Mechanistic Causation|Contrastive Explanation|Functional|Correlation|Pragmatic Approach|No Explanation)\b", response_text)
            all_classifications.extend(classifications)

    # Count the occurrences of each classification
    classification_counts = Counter(all_classifications)

    # Return the most common classification or the first one in case of ties
    if classification_counts:
        most_common_classification = classification_counts.most_common(1)[0][0]
        return [most_common_classification]
    else:
        return ["No Classification Found"]

# Classify sentences with each prompt and save results
for prompt_name, prompt_template in templates.items():
    classified_sentences = []
    for sentence in test_sentences:
        classified_output = classify_sentence_with_prompt(sentence, prompt_template)
        classified_sentences.append(classified_output)
        # Print the sentence, the classification label, and the prompt file name
        print(f"Prompt File: {prompt_name}")
        print(f"Sentence: {sentence}")
        print(f"Classified Output: {classified_output}\n")

    result_df = pd.DataFrame({
        'original': test_sentences,
        'classified': classified_sentences
    })

    output_file_path = os.path.join(directory_path, f'Classifications-{prompt_name}.tsv')
    result_df.to_csv(output_file_path, sep='\t', index=False, header=True, encoding='utf-8')
    print(f"\n==== Results for {prompt_name} saved to {output_file_path} ====")
    print(result_df.head())

