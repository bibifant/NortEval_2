import spacy
import json
import os

nlp = spacy.load("de_core_news_sm")


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def sentence_contains_verb(sentence):
    doc = nlp(sentence)
    return any(token.pos_ in ["VERB", "AUX"] for token in doc)


def get_verbs(sentence):
    doc = nlp(sentence)
    verbs = [token.text for token in doc if token.pos_ in ["VERB", "AUX"]]
    return verbs


def avg_percentage(dataset_points, key):
    return sum(point[key] for point in dataset_points) / len(dataset_points)


def save_contains_verb_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"contains_verb_results": data}, output_file, ensure_ascii=False, indent=2)


def rating(percentage):
    if 50 <= percentage <= 80:
        return "average"
    if percentage > 80:
        return "good"
    if percentage < 50:
        return "bad"


def update_results_file(output_folder, avg_true_percentage):
    rating_value = rating(avg_true_percentage)

    avg_result_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(avg_result_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average value
    existing_data["Results"].append({"Contains verb": {
        "average_percentage_of_correct_verbs_contained": round(avg_true_percentage, 2),
        "average_percentage_of_correct_verbs_contained_rating": rating_value
    }})

    # Update results file
    with open(avg_result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=2)


def run_contains_verb(output_folder):
    print(f"Contains verb is running.")

    json_file_path = os.path.join(output_folder, "bleu_results.json")
    output_file_path = os.path.join(output_folder, "contains_verb_results.json")

    dataset_points = []

    data = load_data(json_file_path)

    for index, data_point in enumerate(data.get('scores', [])):

        response_text = data_point.get('response', '')
        doc = nlp(response_text)

        sentences_with_verb_count = sum(1 for sent in doc.sents if sentence_contains_verb(sent.text))
        total_sentences = len(list(doc.sents))
        percentage_with_verb = (sentences_with_verb_count / total_sentences) * 100
        containing_verbs = get_verbs(response_text)

        dataset_points.append({
            "index": index,
            "response_text_bleu": response_text,
            "percentage_of_sentences_containing_at_least_one_verb": percentage_with_verb,
            "containing_verbs": containing_verbs
        })

    save_contains_verb_results(output_file_path, dataset_points)

    avg_contains_verb = avg_percentage(dataset_points, "percentage_of_sentences_containing_at_least_one_verb")

    update_results_file(output_folder, avg_contains_verb)
