import json
import os
from rouge_score import rouge_scorer
from openai_connection import get_answer, load_model_and_tokenizer


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


def run_rouge(model_name, tokenizer, max_index=2):
    print("ROUGE score is being calculated.")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model, tokenizer = load_model_and_tokenizer(model_name)

    dataset_points = []
    lines = load_data('./datasets/test.json')
    for index, line in enumerate(lines[:max_index]):
        data_point = json.loads(line)

        prompt = f"Fasse den Text auf deutsch zusammen:\n{data_point.get('wiki_sentences')}"

        # Pass both model and tokenizer to get_answer
        response = get_answer(model, tokenizer, prompt)  # Updated this line

        reference = data_point.get("klexikon_sentences")
        reference_str = ' '.join(reference)

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

    avg_rouge1, avg_rouge2, avg_rougeL = calculate_average_rouge_scores(dataset_points)
    return avg_rouge1, avg_rouge2, avg_rougeL



def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines
