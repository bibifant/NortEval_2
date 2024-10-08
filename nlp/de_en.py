import spacy
import json
import os
from langdetect import detect

nlp = spacy.load("de_core_news_sm")


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def calculate_language_percentages(doc):
    english_words = sum(1 for token in doc if token.is_alpha and detect(token.text) == 'en')
    german_words = sum(1 for token in doc if token.is_alpha and detect(token.text) == 'de')
    total_words = sum(1 for token in doc if token.is_alpha)

    # Avoid division by zero
    if total_words == 0:
        english_percentage, german_percentage = 0, 0
    else:
        # Calculate percentage values
        english_percentage = (english_words / total_words) * 100
        german_percentage = (german_words / total_words) * 100

        # Adjustment of the percentage values to a total of 100%
        if english_percentage + german_percentage != 100:
            scale_factor = 100 / (english_percentage + german_percentage)
            english_percentage = english_percentage * scale_factor
            german_percentage = german_percentage * scale_factor

    return round(english_percentage, 2), round(german_percentage, 2)


def calculate_average_percentage(dataset_points, key):
    return sum(point[key] for point in dataset_points) / len(dataset_points)


def rating(percentage):
    if 50 <= percentage <= 80:
        return "average"
    if percentage > 80:
        return "good"
    if percentage < 50:
        return "bad"


def update_results_file(output_folder, avg_english_percentage, avg_german_percentage):
    rating_value = rating(avg_german_percentage)

    result_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(result_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average values
    existing_data["Results"].append({"Percentage of German & English words": {
        "average_percentage_of_english_words": round(avg_english_percentage, 2),
        "average_percentage_of_german_words": round(avg_german_percentage, 2),
        "average_percentage_of_german_words_rating": rating_value
    }})

    # Update results file
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)


def save_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"results": data}, output_file, ensure_ascii=False, indent=2)


def run_language_percentage(output_folder):
    print(f"Percentage of German & English words is being assessed.")
    json_file_path = os.path.join(output_folder, "bleu_results.json")
    output_file_path = os.path.join(output_folder, "language_percentage_results.json")

    dataset_points = []

    data = load_data(json_file_path)

    for index, data_point in enumerate(data.get('scores', [])):

        response_text = data_point.get('response', '')
        doc = nlp(response_text)

        # Determine the main language of the text
        main_language = detect(response_text)

        english_percentage, german_percentage = calculate_language_percentages(doc)

        dataset_points.append({
            "index": index,
            "response_text_bleu": response_text,
            "main_language_part_of_the_text": main_language,
            "english_part_of_speech": english_percentage,
            "german_part_of_speech": german_percentage
        })

    save_results(output_file_path, dataset_points)

    avg_english_percentage = calculate_average_percentage(dataset_points, "english_part_of_speech")
    avg_german_percentage = calculate_average_percentage(dataset_points, "german_part_of_speech")

    update_results_file(output_folder, avg_english_percentage, avg_german_percentage)
