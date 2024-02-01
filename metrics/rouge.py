import json
import os.path
from rouge_score import rouge_scorer
from script.azure_openai_connection import get_answer


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def round_rouge_scores(scores):
    return {
        'rouge1': round(scores['rouge1'].fmeasure, 2),
        'rouge2': round(scores['rouge2'].fmeasure, 2),
        'rougeL': round(scores['rougeL'].fmeasure, 2),
    }


def calculate_average_rouge_scores(dataset_points):
    rouge1_scores = [point["rouge1_scores"] for point in dataset_points]
    rouge2_scores = [point["rouge2_scores"] for point in dataset_points]
    rougeL_scores = [point["rougeL_scores"] for point in dataset_points]

    avg_rouge1 = round(sum(rouge1_scores) / len(rouge1_scores), 2)
    avg_rouge2 = round(sum(rouge2_scores) / len(rouge2_scores), 2)
    avg_rougeL = round(sum(rougeL_scores) / len(rougeL_scores), 2)

    return avg_rouge1, avg_rouge2, avg_rougeL


def rating_rouge1(score):
    if 0.2 <= score <= 0.3:
        return "moderate"
    if score > 0.3:
        return "good"
    if score < 0.2:
        return "low"


def rating_rouge2(score):
    if 0.05 <= score <= 0.15:
        return "moderate"
    if score > 0.15:
        return "good"
    if score < 0.05:
        return "low"


def rating_rougeL(score):
    if 0.1 <= score <= 0.2:
        return "moderate"
    if score > 0.2:
        return "good"
    if score < 0.1:
        return "low"


def update_results_file(output_folder, avg_rouge1, avg_rouge2, avg_rougeL):
    rating_rouge1_value = rating_rouge1(avg_rouge1)
    rating_rouge2_value = rating_rouge2(avg_rouge2)
    rating_rougeL_value = rating_rougeL(avg_rougeL)

    result_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(result_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average values
    existing_data["Results"].append({
        "average_rouge1": avg_rouge1,
        "average_rouge1_rating": rating_rouge1_value,
        "average_rouge2": avg_rouge2,
        "average_rouge2_rating": rating_rouge2_value,
        "average_rougeL": avg_rougeL,
        "average_rougeL_rating": rating_rougeL_value
    })

    # Update results file
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)


def save_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"results": data}, output_file, ensure_ascii=False, indent=2)


def run_rouge(output_folder):
    # Initialize rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    json_file_path = "./dataset/test.json"
    output_file_path = os.path.join(output_folder, "rouge_results.json")

    dataset_points = []

    lines = load_data(json_file_path)

    # Calculate Rouge scores for the first 2 data points
    for index, line in enumerate(lines[:2]):
        data_point = json.loads(line)

        prompt = f"Fasse den Text auf deutsch zusammen:\n{data_point.get('wiki_sentences')}"

        response = get_answer(prompt)
        reference = data_point.get("klexikon_sentences")
        # Convert into Strings
        reference_str = ' '.join(reference)

        # Calculate Rouge scores
        scores = scorer.score(reference_str, response)

        rounded_scores = round_rouge_scores(scores)

        dataset_point = {
            "index": index,
            "prompt": prompt,
            "response": response,
            "reference": reference_str,
            "rouge1_scores": rounded_scores['rouge1'],
            "rouge2_scores": rounded_scores['rouge2'],
            "rougeL_scores": rounded_scores['rougeL']
        }

        dataset_points.append(dataset_point)

    # Save Rouge Scores
    save_results(output_file_path, dataset_points)

    # Calculate average Rouge scores
    avg_rouge1, avg_rouge2, avg_rougeL = calculate_average_rouge_scores(dataset_points)

    # Save average Rouge Scores
    update_results_file(output_folder, avg_rouge1, avg_rouge2, avg_rougeL)
