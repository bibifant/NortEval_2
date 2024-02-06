import json
import os

import Levenshtein
from azure_openai_connection import get_answer

prompt_template = "Categorize which sentiment the example word contains: "
prompt_additional_instructions = "Respond in all lower caps and in a single word."
ds_json_file_path = "datasets/sentiment_analysis_ds.json"


def load_data(ds_json_file_path):
    with open(ds_json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get("Wörter", [])


def generate_prompt(prompt_template, word):
    return f"{prompt_template} '{word}', 'very positive', 'positive', 'neutral', 'negative' or 'very negative'.{prompt_additional_instructions}"


def create_json_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def check_sentiment_match_exact(response_sentiment, reference_sentiment, allowed_distance=1):
    # If response_sentiment is a list, consider the first element
    if isinstance(response_sentiment, list):
        response_sentiment_lower = response_sentiment[0].lower()
    else:
        response_sentiment_lower = str(response_sentiment).lower()

    # Convert reference_sentiment to lowercase for case-insensitive comparison
    reference_sentiment_lower = reference_sentiment.lower()

    # Check if the response sentiment exactly matches the reference sentiment
    exact_match = response_sentiment_lower == reference_sentiment_lower

    # Check if the Levenshtein distance is within the allowed threshold
    distance = Levenshtein.distance(response_sentiment_lower, reference_sentiment_lower)
    within_distance_threshold = distance <= allowed_distance

    # Consider it a match if either the exact match or within_distance_threshold is true
    return exact_match or within_distance_threshold


def check_sentiment_match_in_category(response_sentiment, reference_sentiment):
    # Define sentiment categories
    negative_categories = ['negative', 'very negative']
    positive_categories = ['neutral', 'positive', 'very positive']
    neutral_categories = ['neutral']

    # If response_sentiment is a list, consider the first element
    if isinstance(response_sentiment, list):
        response_sentiment_lower = response_sentiment[0].lower()
    else:
        response_sentiment_lower = str(response_sentiment).lower()

    # Convert reference_sentiment to lowercase for case-insensitive comparison
    reference_sentiment_lower = reference_sentiment.lower()

    # Check if the model recognized the general sentiment correctly
    if reference_sentiment_lower in negative_categories:
        return response_sentiment_lower in negative_categories
    elif reference_sentiment_lower in positive_categories:
        return response_sentiment_lower in positive_categories
    elif reference_sentiment_lower in neutral_categories:
        return response_sentiment_lower in neutral_categories
    else:
        return False


def update_results_file(output_folder, percentage_exact_matches, percentage_category_matches,
                        result_category_exact_match, result_category_category_match):
    result_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(result_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average values
    existing_data["Results"].append({'Sentiment analysis': {
        'percentage_exact_sentiment_recognized': percentage_exact_matches,
        'result_category_exact_recognition': result_category_exact_match,
        'percentage_sentiment_category_recognized': percentage_category_matches,
        'result_category_sentiment_category_recognition': result_category_category_match
    }})

    # Update results file
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)


def save_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"results": data}, output_file, ensure_ascii=False, indent=2)


def run_sentiment_analysis(output_folder):
    # Load data from JSON file
    words_data = load_data(ds_json_file_path)
    #create detailed results file for sentiment analysis
    output_file_path = os.path.join(output_folder, "sentiment_analysis_results.json")

    results = []

    correct_exact_matches = 0
    correct_category_matches = 0

    for word_info in words_data:
        word = word_info.get("Wort", "")
        reference_sentiment = word_info.get("Sentiment",
                                            "")

        # Generate prompt and get sentiment analysis
        prompt = generate_prompt(prompt_template, word)
        sentiment = get_answer(prompt)

        # Check if the sentiment matches the reference sentiment exactly or in category
        exact_match = check_sentiment_match_exact(sentiment, reference_sentiment)
        category_match = check_sentiment_match_in_category(sentiment, reference_sentiment)

        # print(f"The sentiment of the word '{word}' is: {sentiment}")
        # print(f"Reference sentiment: {reference_sentiment}")
        # print(f"Sentiments exact match: {exact_match}")
        # print(f"Sentiments match in category: {category_match}")

        # Collect results
        result_entry = {
            'word': word,
            'prompt': prompt,
            'response': sentiment,
            'reference_sentiment': reference_sentiment,
            'exact_match': exact_match,
            'category_match': category_match
        }
        results.append(result_entry)
        # Count correct matches
        if exact_match:
            correct_exact_matches += 1
        if category_match:
            correct_category_matches += 1

    total_entries = len(words_data)
    percentage_exact_matches = round((correct_exact_matches / total_entries) * 100, 2)
    percentage_category_matches = round((correct_category_matches / total_entries) * 100, 2)

    #
    # print(f"\nPercentage of correct exact matches: {percentage_exact_matches}%")
    # print(f"Percentage of correct category matches: {percentage_category_matches}%")

    def categorize_results(percentage_matches, threshold_good=85, threshold_bad=65):
        if percentage_matches >= threshold_good:
            return "good"
        elif percentage_exact_matches >= threshold_bad:
            return "average"
        else:
            return "bad"

    result_category_exact_match = categorize_results(percentage_exact_matches, 70, 50)
    result_category_category_match = categorize_results(percentage_category_matches)
    #
    # print(f"\nExact Match Result Category: {result_category_exact_match}")
    # print(f"\nCategory Match Result Category: {result_category_category_match}")

    # Save average results
    update_results_file(output_folder, percentage_exact_matches, percentage_category_matches,
                        result_category_exact_match, result_category_category_match)

    # Create JSON file from the results
    create_json_file(results, output_file_path)
