import spacy
import os
import json

nlp = spacy.load("de_core_news_sm")


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def calculate_percentage(doc, nouns, words_lower_case):
    nouns_capitalized = [word.capitalize() for word in nouns]

    # Calculate percentage values
    percentage_nouns = min(((sum(1 for token in doc if token.text in nouns_capitalized) / len(nouns_capitalized)) * 100) if len(nouns_capitalized) > 0 else 0, 100)
    percentage_low = ((sum(1 for token in doc if token.text.isalpha() and token.text.lower() not in nouns and not token.is_sent_start and token.text.islower()) / len( words_lower_case)) * 100) if len(words_lower_case) > 0 else 0
    percentage_start_of_sentence = (sum(1 for token in doc if token.is_sent_start and token.text.istitle()) / len(list(doc.sents))) * 100
   
    total_percentage = (percentage_nouns + percentage_low + percentage_start_of_sentence) / 3
    return round(total_percentage, 2)


def save_results(output_file_path, dataset_points):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"results": dataset_points}, output_file, ensure_ascii=False, indent=2)


def calculate_average_percentage(dataset_points, key):
    return sum(point[key] for point in dataset_points) / len(dataset_points)


def rating(percentage):
    if 50 <= percentage <= 80:
        return "average"
    if percentage > 80:
        return "good"
    if percentage < 50:
        return "bad"


def update_results_file(output_folder, avg_upper_lower_case):
    rating_value = rating(avg_upper_lower_case)

    avg_results_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(avg_results_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average value
    existing_data["Results"].append({"Correct upper lower case": {
        "average_of_correct_letter_case": round(avg_upper_lower_case, 2),
        "average_of_correct_letter_case_rating": rating_value
    }})

    # Update results file
    with open(avg_results_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)


def run_upper_lower_case(output_folder):
    print(f"Upper Lower Case is running.")
    json_file_path = os.path.join(output_folder, "bleu_results.json")
    output_file_path = os.path.join(output_folder, "upper_lower_case_results.json")

    dataset_points = []

    data = load_data(json_file_path)

    for index, data_point in enumerate(data.get('scores', [])):

        response_text = data_point.get('response', '')
        doc = nlp(response_text)

        text_low = response_text.lower()
        nouns = [token.text for token in nlp(text_low) if token.pos_ in ('NOUN', 'PROPN')]
        words_lower_case = [token.text for token in doc if token.text.isalpha() and token.text.lower() not in nouns and not token.is_sent_start]

        total_percentage = calculate_percentage(doc, nouns, words_lower_case)

        dataset_points.append({
            "index": index,
            "response_text_bleu": response_text,
            "correct_upper_and_lower_case": total_percentage
        })

    save_results(output_file_path, dataset_points)

    avg_upper_lower_case = calculate_average_percentage(dataset_points, "correct_upper_and_lower_case")

    update_results_file(output_folder, avg_upper_lower_case)
