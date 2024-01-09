# notal-llm: German LLM Evaluation Module

## Overview

This Python module is designed to evaluate the output quality of Language Models (LLMs) with a focus on generating high-quality German text. The evaluation utilizes three key metrics: Perplexity, BLEU, and ROUGE. Additionally some NLP-Tests are also performed (de_en, functionGK and gebeugtes_verb).

## Installation

//todo

```bash
pip install -r requirements.txt
```

## Usage:

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

run_rouge()

import json
from rouge_score import rouge_scorer
import sys
sys.path.append('script')
from script.azure_openai_connection import get_answer

def run_rouge():
    This takes datapoints from the dataset and calculated three rouge scores for each entry. It creates a txt file with the scores and the average scores.

## functionGK : Capitalization Test
run_function_gk_test(text_example_gk)

import spacy

global variable
nlp = spacy.load("de_core_news_sm")

def upper_lower_case(text):
    This analyzes if an input sentence contains upper and lower case. 

def run_function_gk_test(text_example_gk):
    This prints out the results.

## gebeugtes_verb : Verb Conjugation Test

run_gebeugtes_verb_test(saetze)

import spacy

global variable
nlp = spacy.load("de_core_news_sm")

def get_number(sentence):
    This determines the number of the verbs in the sentence and returns the number.

def get_person(sentence):
    This determines the person of the verbs in the sentence and returns the person.

def get_nomen_numerus(sentence):
    This determines the number of the nouns in the sentence and returns the number. 

def enthält_gebeugtes_verb(satz):
    Returns TRUE, if sentence contains a conjugated verb, else FALSE

def run_gebeugtes_verb_test(saetze):
    This iterates through a list of sentences.

## de_en Language Percentage Check

run_de_en_test(text)

import spacy
from langdetect import detect

global variable
nlp = spacy.load("de_core_news_sm")

def language_percentage(text):
    This calculates the percentage of a certain language in a sentence (German, English).

def run_de_en_test(text):
    This executes language_percentage(text).



