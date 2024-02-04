import spacy
import json
import os
from langdetect import detect
from script.azure_openai_connection import get_answer

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


def update_results_file(output_folder, avg_english_percentage, avg_german_percentage):
    result_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(result_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average values
    existing_data["Results"].append({
        "average percentage of english words": round(avg_english_percentage, 2),
        "average percentage of german words": round(avg_german_percentage, 2)
    })

    # Update results file
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)


def save_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"results": data}, output_file, ensure_ascii=False, indent=2)


def run_language_percentage(output_folder):
    json_file_path = "./datasets/nlp_dataset.json"
    output_file_path = os.path.join(output_folder, "language_percentage_results.json")

    dataset_points = []

    data = load_data(json_file_path)

    for index, data_point in enumerate(data):
        prompt = f"Beantworte folgende Frage auf deutsch: {data_point.get('Frage')}"

        # Response LLM
        response = get_answer(prompt)
        response_str = ' '.join(sentence + "." for sentence in response)

        doc = nlp(response_str)

        # Determine the main language of the text
        main_language = detect(response_str)

        english_percentage, german_percentage = calculate_language_percentages(doc)

        dataset_points.append({
            "index": index,
            "prompt": prompt,
            "response": response_str,
            "main language part of the text": main_language,
            "english part of speech": english_percentage,
            "german part of speech": german_percentage
        })

    save_results(output_file_path, dataset_points)

    avg_english_percentage = calculate_average_percentage(dataset_points, "english part of speech")
    avg_german_percentage = calculate_average_percentage(dataset_points, "german part of speech")

    update_results_file(output_folder, avg_english_percentage, avg_german_percentage)


