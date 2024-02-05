# nortal-llm: German LLM Evaluation Module

## Overview

This Python module is designed to evaluate the output quality of Language Models (LLMs) with a focus on generating high-quality German text. The evaluation utilizes three key metrics: Perplexity, BLEU, and ROUGE. Additionally some NLP-Tests are also performed (de_en, upper_lower_case and contains_verb).

## Installation

```bash
pip install -r requirements.txt
```

## Usage:

# Metrics

## BLEU
This script implements the BLEU metric. It measures the similarity between a machine-generated translation and one or more reference translations provided by humans.
In this implementation we provide an English source text to be translated into German and compare the LLMs output to a reference translation from a dataset (//add dataset).

json_file_path = "dataset/bleu_dataset.json"
run_bleu_test_on_json_dataset(json_file_path)

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def run_bleu_test_on_json_dataset(json_file_path):
    This generates a prompt with text to be translated (from dataset) and instruction to translate it into German. The response is then compared to a reference translation with the BLEU metric and a score is calculated.
    The higher the BLUE score, the better.

## Perplexity
dataset_path = "WikiQA-train.txt"
run_perplexity_test(dataset_path)

import nltk
from nltk import FreqDist
from nltk.lm import MLE
from nltk.util import bigrams
import math

def load_and_preprocess_dataset(dataset_path):
    This is a necessary to prepare the data for further processing.

def tokenize_text(text):
    This breaks the text down into tokens.

def calculate_perplexity(tokens, ngram_order=2):
    This calculates the perplexity score for the text (datapoint) from the dataset in accordance with the previous tokenization and returns a score. 

def run_perplexity_test(dataset_path):
    This prints the result.

## Rouge

#### The Rouge metric is used to evaluate generated text summaries of the LLM by measuring the alignment between the generated and reference texts based on n-gram matches (Rouge-1, Rouge-2) and the longest common subsequence (Rouge-L).

import json
import os.path
from rouge_score import rouge_scorer
from script.azure_openai_connection import get_answer


def load_data(json_file_path):
    This reads the json data from "test.json" file and returns the loaded data

def round_rouge_scores(scores):
    This rounds the rouge scores to two decimal places.

def calculate_average_rouge_scores(dataset_points):
    This function calculates the average Rouge scores for Rouge-1, Rouge-2, and Rouge-L based on the list of data points, 
    each containing these scores for different text summaries.

def rating_rouge1(score):
    This function evaluates the Rouge1 score in three categories: "low", "moderate" and "good".

def rating_rouge2(score):
    This function evaluates the Rouge2 score in three categories: "low", "moderate" and "good".

def rating_rougeL(score):
    This function evaluates the RougeL score in three categories: "low", "moderate" and "good".

def update_results_file(output_folder, avg_rouge1, avg_rouge2, avg_rougeL):
    This updates the "avg_results.json" file in the "results" folder with new results by reading existing data, 
    adding the new data and writing the updated data back to the file.

def save_results(output_file_path, data):
    This writes the data to the "rouge_results.json" file and saves it there.

def run_rouge():
    This initializes the Rouge scorer, processes a dataset of German text and  generates summaries of "candidate_summary" using the LLM. 
    It then calculates Rouge scores for the first 2 entries in the dataset by comparing the candidate summaries to the reference summaries.
    The scores, prompts and responses are then saved into the json file.


# NLP (Natural Language Processing) Methods

## upper lower case : Capitalization Test

#### This module calculates the percentage of correct capitalization of the generated text from the connected API.

import spacy
import os
import json
from script.azure_openai_connection import get_answer

nlp = spacy.load("de_core_news_sm")
    This loads the german language model from spacy

def load_data(json_file_path):
    This reads the json data from "npl_dataset.json" file and returns the loaded data

def calculate_percentage(doc, nouns, words_lower_case):
    This calculates the percentage of capitalized nouns, lowercase words not in the specified list, and title-cased words at the start of sentences. 
    It averages these three percentages and rounds the result to two decimal places.

def save_results(output_file_path, dataset_points):
    This writes the data to the "upper_lower_case_results.json" file and saves it there.

def calculate_average_percentage(dataset_points):
    This function calculates the average percentage of correct usage of upper and lower case letters across the dataset and rounds the result to two decimal places.

def update_results_file(output_folder, avg_upper_lower_case):
    This updates the "avg_results.json" file in the "results" folder with new results by reading existing data, 
    adding new data and writing the updated data back to the file.

def run_upper_lower_case(output_folder):
    This processes the dataset and responses of the api, calculates the percentage of correct usage of upper and lower case letters in the answers, saves individual results to the "upper_lower_case_results.json" file, 
    and updates the "avg_results.json" file with the calculated average percentage.



## contains_verb : Verb  Test

#### This module checks whether a verb is contained in a sentence.

import spacy
import json
import os
from script.azure_openai_connection import get_answer

nlp = spacy.load("de_core_news_sm")
    This loads the German language model from spacy

def load_data(json_file_path):
    This reads the json data from "npl_dataset.json" file and returns the loaded data

def sentence_contains_verb(sentence):
    This checks whether the sentence contains a verb or an auxiliary verb 

def get_verbs(sentence):
    This analyzes a sentence using spacy and returns a list of verbs and auxiliary verbs in the sentence

def avg_percentage(dataset_points, key):
    This calculates the average value whether a conjugated verb is included.

def save_contains_verb_results(output_file_path, data):
    This writes the data to the "contains_verb_results.json" file and saves it there.

def update_results_file(output_folder, avg_true_percentage):
    This updates the "avg_results.json" file in the "results" folder with new results by reading existing data, 
    adding new data and writing the updated data back to the file.

def run_contains_verb(output_folder):
    This analyzes each response to calculate the percentage of sentences containing at least one verb. It also extracts the verbs from the responses.It then saves the results including the response sentences, percentage with verbs, 
    and the verbs themselves into the json file.


## de_en: Language Percentage Check

#### This module identifies the main language of the generated answers,using Spacy for linguistic analysis and the langdetect library.

import spacy
import json
import os
from langdetect import detect
from script.azure_openai_connection import get_answer

nlp = spacy.load("de_core_news_sm")
    This loads the German language model from spacy

def load_data(json_file_path):
    This reads the json data from "npl_dataset.json" file and returns the loaded data

def calculate_language_percentages(doc): 
    This calculates the percentage of words in English and German and rounds the result to two decimal places.

def calculate_average_percentage(dataset_points, key):
    This function calculates the average percentage of English and German words for all datapoints across the dataset.

def update_results_file(output_folder, avg_english_percentage, avg_german_percentage):
    This updates the "avg_results.json" file in the "results" folder with new results by reading existing data, 
    adding new data and writing the updated data back to the file.

def save_results(output_file_path, data):
    This writes the data to the "language_percentage_results.json" file and saves it there.

def run_language_percentage(output_folder):
    This processes the dataset and responses of the api and calculates the percentage of English and German words in the generated answers.
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






