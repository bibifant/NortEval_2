import json
from bleu.utils import fetch_dataset_from_api
from script.azure_openai_connection import get_simple_translation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
BLEU_API = os.getenv("BLEU_DATASET_API_URL") # dataset link: https://huggingface.co/datasets/abidlabs/test-translation-dataset
# create output file path
output_file_path = "results/bleu_with_online_dataset_results.json"


"""
This file uses BLEU to calculate scores for short text.
The BLEU score is calculated in the following steps:
1. Fetch data from the API or the local dataset (all methods can be found in the 'utils' file).
2. Filter the raw dataset to get only the needed data and return it as a list, for example: dataset = [{'en': '...', 'de': '...'}, ...].
3. The text value of 'de' is used as the human reference, and predictions are the text values of 'en', which need to be translated to German.
4. Use OpenAI to translate the text of 'en' to German and use 'sentence_bleu()' to calculate the score.
5. Reformat the score to display it as a percentage (e.g., 0.80).
6. Calculate the average score.
7. Save all the results as a JSON file.
"""


def calculate_bleu():
    # Fetch dataset from API URL
    data = fetch_dataset_from_api(BLEU_API)

    # Filter dataset
    dataset = extract_translations_input_translation(data)

    # Initialize total score and count of successful BLEU score calculations
    total_bleu_score, count = 0, 0
    results = {"scores": []}

    for i, data_point in enumerate(dataset):
        # Translate 'en' prediction to German using OpenAI
        prediction = get_simple_translation(data_point.get("en", ""))
        # 'de' is used as a reference
        reference = data_point.get("de", "")

        # Check if prediction or reference is None
        if not prediction or not reference:
            print(f"Incorrect datapoint {i + 1}: Missing human reference or prediction.")
        else:
            # Calculation of the BLEU score for the prediction compared to the human reference
            bleu_score = sentence_bleu([prediction.split()], reference.split(),
                                       smoothing_function=SmoothingFunction().method1,
                                       weights=(1, 0))  # Calculate BLEU with uni-gram
            total_bleu_score += bleu_score  # Set total BLEU score

            # Formatting the BLEU score to two decimal places after the decimal point
            formatted_bleu_score = "{:.2f}".format(bleu_score)

            # Save the results as a dictionary
            results["scores"].append({
                "prediction_index": i + 1,
                "bleu_score": formatted_bleu_score
            })
            count += 1

    # Average BLEU score for the entire dataset
    average_bleu_score = total_bleu_score / count if count > 0 else 0
    # Formatting the average BLEU score to two decimal places after the decimal point
    formatted_bleu_average_score = "{:.2f}".format(average_bleu_score)
    results["count"] = count
    results["average_bleu_score"] = formatted_bleu_average_score

    # Saving the results as JSON
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=2)


# Extract translations from JSON data fetched from the dataset API
def extract_translations_input_translation(data):
    dataset = []
    for row in data['rows']:
        key_row = row.get('row', {})
        de_content = key_row.get('Translation', '')
        en_content = key_row.get('Input', '')
        if de_content and en_content:
            dataset.append({'de': de_content, 'en': en_content})
    return dataset
