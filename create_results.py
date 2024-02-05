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
            "BLEU": "BLEU evaluates the quality of machine translations. Bleu scores range from 0 to 1. The higher the BLEU score the better the quality of the machine translation.",
            "Perplexity": "Perplexity is a measure of the predictive uncertainty of a language model. A lower value indicates a higher prediction accuracy of the model. A value close to 50 is excellent.",
            "Naturalness": "This score is calculated using perplexity. It assesses the fluency and coherence of the generated text. A lower score usually indicates text that is more natural and easy to read. A score below 50 is excellent.",
            "Semantic Similarity": "This score evaluates how relevant and contextually aligned the model's response is to the given prompt. The score ranges from 0 to 1. A higher score implies that in meaning the generated text is closely related to the prompt.",
            "Keywords in Response": "This score measures the presence of key concepts in the response. The score ranges from 0 to 1. A higher score signifies that the response encompasses key elements from the prompt.",
            "Sentiment Analysis": "Sentiment analysis is a natural language processing task that involves determining and categorizing the emotional tone in a piece of text. \nCategories used: very positive, positive, neutral, negative, very negative. Scale ranges 0 - 100%. \nThe higher the percentage, the more sentiments have been identified correctly by the model. \nFor more information see https://confluence.de.nortal.com/display/HTWSTUD/Project+Approach#ProjectApproach-SentimentAnalysis"
        },
        "Results": []
    }

    with open(file_name, mode='w', encoding='utf-8') as file:
        json.dump(data_structure, file, ensure_ascii=False, indent=4)
    print(f"The folder {folder_name} is being created.")

    return folder_name
