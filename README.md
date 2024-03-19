# Contents of this repository

This repository contains all the data used for testing and finetuning the Cohere API Classifier models mentioned in the paper, as well as an example promp to illustrate how each sentence was tested with the GPT-4 architecture.

The manually annotated sentence seed is also freely available to use, with the inclusion of each sentence category, topic domain and consensus score.

An update with the full coding pipeline for each model should be on the way, as soon as everything is neatly compiled in `.ipynb` notebooks for ease of understanding.

### Overall folder organization

    Explanation_Datasets
    ├── finetune_sets
    │   ├── expl_tune_binary.tsv               # Data for finetuning of a binary classifier.
    │   └── expl_tune_multiclass.tsv           # Data for finetuning a multiclass classifier.
    ├── test_sets
    │   └── pmc_test_set.csv                   # Set of approx 3670 sentences from the PMC Open Source corpus 008 split.
    ├── annotated_sentence_seed.csv            # Sentence seed of explanatory sentences, annotated and user-reviewed.
    ├── prompt_template.txt                    # Example template used for prompting GPT-4 as a classifier.
    └── README.md
