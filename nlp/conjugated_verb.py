#  Check whether a conjugated verb is contained in the sentence. As the verb is the anchor of a German sentence,
#  one must be included.

import spacy
import json
import os
from script.azure_openai_connection import get_answer

nlp = spacy.load("de_core_news_sm")


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_number(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            number_list = token.morph.get('Number')
            if number_list:
                number = number_list[0]
                return number


def get_nouns_numerus(sentence):
    doc = nlp(sentence)

    # Check if the word is a plural noun
    for token in doc:
        if token.pos_ == "NOUN":
            numerus_list = token.morph.get("Number")
            numerus = numerus_list[0]
            return numerus


def get_verbs(sentence):
    doc = nlp(sentence)
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    return verbs


def avg_percentage(dataset_points):
    if not dataset_points:
        return 0

    # count the number of true values
    true_count = sum(1 for point in dataset_points if point["contains_conjugated_verb"])

    # calculate average percentage
    average = (true_count / len(dataset_points)) * 100

    return average


def save_conjugated_verb_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"conjugated_verb_results": data}, output_file, ensure_ascii=False, indent=2)


def update_results_file(output_folder, avg_true_percentage):
    result_file_path = os.path.join(output_folder, "avg_results.json")

    # check if the file exists already
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as result_file:
            existing_data = json.load(result_file)
    else:
        # if file exists, initialize the existing_data as empty dictionary
        existing_data = {"Results": []}

    # add the average percentage and round after 2 decimal point
    existing_data["Results"].append({
        "average contains conjugated verbs": round(avg_true_percentage, 2),
    })

    # write the updated data back to the file
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=2)



def run_conjugated_verb(output_folder):

    json_file_path = "./dataset/nlp_dataset.json"
    output_file_path = os.path.join(output_folder, "conjugated_verb_results.json")

    dataset_points = []
    data = load_data(json_file_path)

    for index, data_point in enumerate(data):

        prompt = f"Beantworte folgende Frage auf deutsch: {data_point.get('Frage')}"
        response = get_answer(prompt)
        doc = nlp(response)
        contains_conjugated_verb = False
        conjugated_verbs = get_verbs(response)

        if conjugated_verbs:
            contains_conjugated_verb = True

        for token in doc:
            if token.pos_ == "VERB":
                if get_number(response) == get_nouns_numerus(response) and get_number(response) == "Sing":
                    contains_conjugated_verb = True
                    break



        dataset_points.append({
            "index": index,
            "sentence": response,
            "contains_conjugated_verb": contains_conjugated_verb,
            "conjugated_verbs": conjugated_verbs,

        })

        # save the results
        save_conjugated_verb_results(output_file_path, dataset_points)

    # calculate the average
    avg_true_percentage = avg_percentage(dataset_points)

    # updating the results file with the average values
    update_results_file(output_folder, avg_true_percentage)
