# Nortal LLM: German LLM Evaluation tool

## Overview

This Python module is designed for evaluating the output quality of Language Models (LLMs), with a primary focus on generating high-quality German text.
The evaluation relies on two key metrics: BLEU and ROUGE. 
Additionally, several NLP tests are conducted to comprehensively evaluate the model's performance. 

These NLP tests include:

1. **Natural language quality assessor:** This test uses semantic similarity, key word extractions and perplexity analysis to assess the quality of the model's response in context. 
2. **Hate speech detection:** This script evaluates the model's ability to detect hate speech within provided prompts.
3. **Sentiment analysis:** This module categorizes text into positive, neutral, or negative sentiment to discern its overall tone.
2. **Additional tests:** The module also performs several other tests, such as case sensitivity analysis (upper/lower case), verb presence detection (contains_verb) and language percentage check(de_en).

## Installation

To install Nortal LLM, please follow these steps:

### 1. Clone the Repository

First, clone the repository to your local machine:

   ```bash
   git clone https://gitlab.rz.htw-berlin.de/Christina.Gottschalk/nortal-llm.git
   cd nortal-llm
   ```

### 2. Install the Package

Install the package and its dependencies with:

   ```bash
   python setup.py install
   ```

This command will install all necessary dependencies listed in the `setup.py` file.

```
pip install -r requirements.txt
```

### 3. Create .env File

Create a `.env` file in your project directory with the following content:

```
AZURE_OPENAI_KEY="Your_key"
AZURE_OPENAI_ENDPOINT="Your_endpoint"
```

Replace `"Your_key"` with your Azure OpenAI API key and `"Your_endpoint"` with your Azure OpenAI endpoint URL. Make sure to keep this file secure and not expose your credentials publicly.


### 4. Download spaCy Models

   After installing the package, you need to download the required spaCy models for German:

   ```bash
   python -m spacy download de_core_news_lg
   python -m spacy download de_core_news_sm
   ```

   This step is necessary to ensure functionality of some of the NLP scripts.

## Usage:

# Metrics

## BLEU
This script implements the BLEU metric. It measures the similarity between a machine-generated translation and one or more reference translations provided by humans.
In this implementation we provide an English source text to be translated into German and compare the LLMs output to a reference translation from a dataset (//add dataset).

   ```bash
  json_file_path = "dataset/bleu_dataset.json"
run_bleu_test_on_json_dataset(json_file_path)
   ```


import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def run_bleu_test_on_json_dataset(json_file_path):
    This generates a prompt with text to be translated (from dataset) and instruction to translate it into German. The response is then compared to a reference translation with the BLEU metric and a score is calculated.
    The higher the BLUE score, the better.



## Rouge
The Rouge metric is used to evaluate generated text summaries of the LLM by measuring the alignment between the generated and reference texts based on n-gram matches (Rouge-1, Rouge-2) and the longest common subsequence (Rouge-L). With our Rouge implementation we can assess the extent to which a model's response aligns with this objective. Rouge tests if the model correctly understands the meaning of long input text and is able to generate a more compact version in our target language, German. So, it needs to differenciate elemental from superfluous information in the input text.Therefore the metric can be a good indicator, if a model responds with relevant information.The score ranges from 0 to 1. A higher value indicates a better summary quality.
  
```bash
def run_rouge(output_folder):
```

This function initializes the Rouge scorer, processes a dataset of German text and generates summaries of "candidate_summary" using the LLM.
It then calculates Rouge scores for the first 2 entries in the dataset by comparing the candidate summaries to the reference summaries.
The scores, prompts and responses are then saved into the json file.


# NLP (Natural Language Processing) Methods

## Natural language quality assessor:
This script employs perplexity as a metric to calculate the fluency and coherence of the generated text. 
A lower score typically signifies text that is more natural and easier to read. Therefore, we refer to this metric as the "naturalness score". 
Additionally, the script uses semantic similarity to assess how relevant and contextually aligned the model's response is to the given prompt. 
A higher semantic similarity score indicates that the generated text closely aligns in meaning with the prompt.
Furthermore, the script extracts key words from the prompt and compares them to the response text to evaluate whether the essential elements of the prompt are addressed in the model's response.



## upper lower case : Correct upper lower case Test
This script calculates the percentage of correct upper lower case of BLEUs responses.By using the German model by spaCy we want to find out whether the beginnings of sentences, nouns, titles, salutations and names of the response texts are capitalized correctly and everything else is written in lower case.The higher the percentage, the better the result.

```bash
def run_upper_lower_case(output_folder):
```

This function takes the responses of BLEU, calculates the percentage of correct usage of upper and lower case letters in the answers, saves individual results to the "upper_lower_case_results.json" file,and updates the "avg_results.json" file with the calculated average percentage.


## contains_verb : Verb Test
This script checks whether a verb is contained in the responses of BLEUs, because the verb is considered the anchor of a sentence. A German sentence must contain at least one verb in order to be well-formed.For this test we also use the German model by spaCy to find out whether a verb is contained in a sentence.The higher the percentage, the better the result.

```bash
def run_contains_verb(output_folder):
```

This function analyzes each response of BLEU to calculate the percentage of sentences containing at least one verb. It also extracts the verbs from the responses.It then saves the results including the response sentences, percentage with verbs,and the verbs themselves into the json file.It then updates the "avg_results.json" file with the calculated average percentage.


## de_en: Language Percentage Check
This script identifies the main language of BLEUs responses,using spaCy for linguistic analysis and the langdetect library.We need this test because most LLMs are able to answer in many different languages. We want to make sure that the answer is in German, as we only want to evaluate German results qualitatively.
The higher the percentage for German, the better the result.

```bash
def run_language_percentage(output_folder):
```

This function takes the responses of BLEU and calculates the percentage of English and German words in the answers.
It then saves individual results to the "language_percentage_results.json" file, and updates the "avg_results.json" file with the calculated average percentage.


# Sentiment Analysis
### This script checks if a model can recognize the connotation of test words and categorizes them from very positive to very negative.

def load_data(ds_json_file_path)
    This function loads the sentiment analysis dataset from the specified JSON file path (ds_json_file_path). It returns a list containing word data extracted from the dataset.

def generate_prompt(prompt_template, word)
    Given a base template (prompt_template) and a word, this function generates a prompt for sentiment analysis. The prompt is formatted with the specified word and additional instructions.
    
def create_json_file(data, output_file_path)
    Writes the provided data (data) to a JSON file located at the specified path (output_file_path). This function is responsible for saving data in JSON format.
    
def check_sentiment_match_exact(response_sentiment, reference_sentiment, allowed_distance=1)
    Checks if the predicted sentiment (response_sentiment) matches the reference sentiment (reference_sentiment) either exactly or within the specified Levenshtein distance (allowed_distance).

def check_sentiment_match_in_category(response_sentiment, reference_sentiment)
    Determines if the predicted sentiment (response_sentiment) falls into the same sentiment category as the reference sentiment (reference_sentiment).

def update_results_file(output_folder, percentage_exact_matches, percentage_category_matches, result_category_exact_match, result_category_category_match)
    Updates the average results file with new sentiment analysis results. It takes in percentages of exact matches, percentages of matches within the same category, and result categories for both exact matches and matches within the same category.

def save_results(output_file_path, data)
    Writes the sentiment analysis results (data) to a JSON file located at the specified path (output_file_path). This function is responsible for saving the detailed sentiment analysis results.

def run_sentiment_analysis(output_folder)
    Processes the sentiment analysis dataset, generates prompts for each word, collects sentiment analysis responses, and saves both detailed and average results. The results are stored in the specified output folder (output_folder).






