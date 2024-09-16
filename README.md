# Contents of this repository

This repository contains all the data for finetuning the BERT base / Cohere API / DeBERTa 3 Small classifier models, testing all models mentioned in the paper, and the prompt used for GPT-4 and Llama 3 8B.

The manually annotated sentence seed is also freely available to use; it includes each sentence category, topic domain and consensus score.

The "executables" folder contains the python pipeline for GPT-4 and Llama 3 8B; the Cohere models can be called through API and are free to use, and their model IDs are the following: " dcd55286-f87e-4b6e-ae54-5193826906d9-ft " for the multiclass model, " acfd0475-3a12-4e74-a0fb-0d0979619076-ft " for the binary model.

### Overall folder organization

    Explanation_Datasets
    ├── executables
    │   ├── classification-gpt-4.py            # Executable for the GPT-4 classification API.
    │   └── classification-llama3-8b.py        # Executable for running Llama 3 8B as a classifier.
    ├── finetune_sets
    │   ├── expl_tune_binary.tsv               # Data for finetuning of a binary classifier.
    │   └── expl_tune_multiclass.tsv           # Data for finetuning a multiclass classifier.
    ├── results
    │   ├── bert-base                         
    |   |   ├── b_binary_results.txt             # Results from BERT Base binary classifier. 
    │   |   └── b_multiclass_results.txt         # Results from BERT Base multiclass classifier.
    │   ├── cohere-llm                         
    |   |   ├── binary_results.tsv             # Results from Cohere binary classifier. 
    │   |   └── multiclass_results.tsv         # Results from Cohere multiclass classifier.
    │   ├── deberta-v3-small                         
    |   |   ├── dv3_binary_results.txt             # Results from DeBERTa v3 Small binary classifier. 
    │   |   └── dv3_multiclass_results.txt         # Results from DeBERTa v3 Small multiclass classifier.
    │   ├── gpt-4                         
    |   |   ├── results_t0.txt                 # Results from GPT-4 with template t0.
    |   |   ├── results_t1.txt                 # Results from GPT-4 with template t1.
    │   |   └── results_t2.txt                 # Results from GPT-4 with template t2.
    |   └── llama3-8b                         
    |       ├── results_t0.tsv                 # Results from Llama 3 8B with template t0. 
    |       ├── results_t1.tsv                 # Results from Llama 3 8B with template t1. 
    │       └── results_t2.tsv                 # Results from Llama 3 8B with template t2. 
    ├── templates
    │   ├── gpt4-t0.txt                        # Template t0 for GPT-4.
    |   ├── gpt4-t1.txt                        # Template t1 for GPT-4.
    |   ├── gpt4-t2.txt                        # Template t2 for GPT-4.
    |   ├── llama3-t0.txt                      # Template t0 for Llama 3 8B.
    │   ├── llama3-t1.txt                      # Template t0 for Llama 3 8B.
    │   └── llama3-t2.txt                      # Template t0 for Llama 3 8B.
    ├── test_sets
    │   ├── pmc_test_set.csv                   # Set of approx 3670 sentences from the PMC Open Source corpus 008 split.
    │   └── test-300-sample.txt                # Random sample taken from the bigger test set to evaluate model performance.
    ├── annotated_sentence_seed.csv            # Sentence seed of explanatory sentences, annotated and user-reviewed.
    └── README.md
