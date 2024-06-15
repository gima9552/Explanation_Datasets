import openai
import langchain
import os
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_key=os.environ.get('OPENAI_API_KEY'),
    api_version="2023-05-15"
)

# Initialize Langchain AzureChatOpenAI
chat_model = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment="gec"
)


# Function to read template from a file
def read_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Load templates from files
templates = {
    "light_knowledge": read_template('template_light_knowledge.txt'),
    "no_knowledge": read_template('template_no_knowledge.txt'),
    "proportional_knowledge": read_template('template_proportional_knowledge.txt')
}

# Create PromptTemplates
prompts = {name: PromptTemplate(input_variables=['text'], template=template) for name, template in templates.items()}

# Create LLMChains
chains = {name: LLMChain(llm=chat_model, prompt=prompt, verbose=True) for name, prompt in prompts.items()}


# Function to classify a sentence using a specified chain
def classify_sentence(sentence, chain):
    result = chain.run(sentence)
    return result


# Function to process a text file with sentences, one per line
def process_text_file(input_path, output_path, chain):
    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        line_number = 0
        for line in infile:
            line_number += 1
            sentence = line.strip()
            if sentence:
                category = classify_sentence(sentence, chain)
                print(f"Processing line {line_number}: {sentence}")
                outfile.write(f"{sentence}\t{category}\n")


# Process the text file with each template
for name, chain in chains.items():
    output_file = f'test-300-results_{name}.txt'
    process_text_file('test-300.txt', output_file, chain)
