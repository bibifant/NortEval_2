import json
import os.path
from datetime import datetime


def create_results():
    # Receive the current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    folder_name = f"results_{timestamp}"

    # Check whether the folder exists. If not, create it.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create file name for the average result file
    file_name = os.path.join(folder_name, "avg_results.json")

    data_structure = {
        "Metrics explanations": {
            "Rouge": "Rouge evaluates the ability of an LLM to generate summaries from an input text. The rouge score ranges from 0 to 1. A higher value indicates a better summary quality.",
            "BLEU": "BLEU is a metric for evaluating the quality of machine translations. Bleu score ranges from 0 to 1. A higher BLEU score indicates that the automatic translations matches the reference translations better. for more information: https://confluence.de.nortal.com/display/HTWSTUD/Metrics#Metrics-BLEUundsacreBLEU and https://confluence.de.nortal.com/display/HTWSTUD/Project+Approach#ProjectApproach-BLEUassessesamodel",
            "Perplexity": "Perplexity is a measure of the predictive uncertainty of a language model. A lower value indicates a higher prediction accuracy of the model. A value close to 50 is excellent.",
            "Sentiment Analysis": "Sentiment analysis is a natural language processing task that involves determining and categorizing the emotional tone in a piece of text. \nCategories used: very positive, positive, neutral, negative, very negative. Scale ranges 0 - 100%. \nThe higher the percentage, the more sentiments have been identified correctly by the model. \nFor more information see https://confluence.de.nortal.com/display/HTWSTUD/Project+Approach#ProjectApproach-SentimentAnalysis",
            "Bias Detection": "Bias detection is a Natural Language Processing (NLP) technique that processes input text in the form of comments from human beings based on the provided input prompt.\n It then compares the output with a binary score of 0/1, indicating 'Not Hate Speech' or 'Hate Speech,' derived from the dataset. The categorizations include Hate Speech and Not Hate Speech, and the scoring range spans from 0% (does not match at all) to 100% (perfect match).",
        },
        "Results": []
    }

    with open(file_name, mode='w', encoding='utf-8') as file:
        json.dump(data_structure, file, ensure_ascii=False, indent=4)
    print(f"The folder {folder_name} is being created.")

    return folder_name
