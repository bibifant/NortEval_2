# NORTevAL: German LLM Evaluation tool

## Overview

This Python module is designed for evaluating the output quality of Language Models (LLMs), with a primary focus on generating high-quality German text.
The evaluation relies on two key metrics: BLEU and ROUGE. 
Additionally, several NLP tests are conducted to comprehensively evaluate the model's performance. 

NLP Tests include:

1. **Natural language quality assessor:** This test uses semantic similarity, key word extractions and perplexity analysis to assess the quality of the model's response in context. 
2. **Hate speech detection:** This script evaluates the model's ability to detect hate speech within provided prompts.
3. **Sentiment analysis:** This module categorizes text into positive, neutral, or negative sentiment to discern its overall tone.
2. **Additional tests:** The module also performs several other tests, such as case sensitivity analysis (upper/lower case), verb presence detection (contains_verb) and language percentage check(de_en).

## Installation

To install Nortal LLM and its required dependencies, please follow these steps:

### 1. Clone the Repository

First, clone the repository to your local machine:

   ```bash
   git clone https://gitlab.rz.htw-berlin.de/Christina.Gottschalk/nortal-llm.git
   ```
### 2. Navigate to the Project Directory

  ```bash
   cd nortal-llm
   ```

### 3. Install Dependencies

   ```bash
   pip install -r requirements.txt
   ```
This command will install all the necessary packages specified in the `requirements.txt` file.


### 4. Create .env File

Create a `.env` file in your project directory with the following content:

``` 
AZURE_OPENAI_KEY="Your_key"
AZURE_OPENAI_ENDPOINT="Your_endpoint"
```

Replace `"Your_key"` with your Azure OpenAI API key and `"Your_endpoint"` with your Azure OpenAI endpoint URL. Make sure to keep this file secure and not expose your credentials publicly.


## Usage:

To run the full suite of evaluation metrics and NLP tests on your Language Model, 
use the `main.py` script. 
This script orchestrates the execution of various components, including BLEU and ROUGE metrics, hate speech detection, sentiment analysis, and natural language quality assessment. 
It also runs additional tests such as language percentage check, upper/lower case sensitivity analysis, and verb presence detection.

### Running the Evaluation
Follow these steps to run the evaluation:

1. **Execute main.py:**
Navigate to the directory containing `main.py` and execute the script:

```bash
python main.py
```
2. **Results:**
The script will automatically create an output folder with a timestamp, 
where all the results from the different tests will be saved. 
This includes individual JSON files for each test, along with an aggregated summary of the results.

3. **Review the Results:**
After completion, review the results in the output folder. 
Each test will generate its own set of detailed results, which you can use to assess the performance and capabilities of your Language Model in handling German text.

   
### Understanding the Results

The results will provide insights into various aspects of your Language Model's performance, including:

- **Translation Quality (BLEU & ROUGE):** Evaluates how well the model translates and summarizes texts in German.
- **Hate Speech Detection:** Assesses the model's ability to identify toxic language.
- **Sentiment Analysis:** Examines the model's proficiency in recognizing and categorizing sentiments.
- **Natural Language Quality:** Analyzes the fluency, coherence, and relevance of the model's responses.
- **Language Accuracy:** Checks the correctness of language usage, including upper/lower case sensitivity and verb presence.
- **Language Preference (de_en):** Determines the predominant language (German or English) in the model's responses.


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

## Hate speech detection
In our hate speech detection implementation, we aim to assess the model’s ability to identify problematic language, 
particularly hate speech, in social media comments. 
We utilize data from the GermEval 2021 webcrawl, which includes comments categorized as toxic (1) or non-toxic (0). 
Our implementation prompts the model to classify each comment accordingly and compares its responses to the reference classifications. 
We calculate the average score of correct toxicity recognition and categorize it into good, average, or bad, allowing for adjustments in categorization thresholds.

```bash
def run_hate_speech(output_folder)
```

This function processes a dataset of social media comments, sending each comment to an AI model for classification as toxic or non-toxic. 
It compares the model’s classifications with the reference classifications in the dataset and saves the results to a JSON file.
This test is essential to ensure that our model does not reproduce language that promotes hatred, violence, or discrimination.


## Sentiment Analysis
### This script checks if a model can recognize the connotation of test words and categorizes them from very positive to very negative.

In our sentiment analysis implementation, we assess the connected model’s proficiency in recognizing word connotations to ensure that its responses are free from bias or unintended meanings. 
We accomplish this by prompting the model to categorize test words based on a predefined scale of connotations and comparing its responses to reference connotations. 
We evaluate the responses using two methods: exact matching and category matching. 
The former checks if the response matches the reference exactly, while the latter evaluates if the response correctly captures the general connotation
This allows for a more nuanced interpretation. 
Additionally, we normalize the response data using Levenshtein distance to account for occasional lexical errors. 
This approach enables us to tailor the evaluation criteria to specific use cases and ensure the model’s responses align with desired standards.

```bash
def run_sentiment_analysis(output_folder) 
```
This function processes the sentiment analysis dataset, generates prompts for each word, collects sentiment analysis responses, 
and saves both detailed and average results. 
The results are stored in the specified output folder (output_folder).

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

