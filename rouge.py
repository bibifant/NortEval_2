import json
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Pfad JSONL-Datei
jsonl_file_path = "de_test.jsonl"

# Daten aus JSONL-Datei lesen
with open(jsonl_file_path, 'r', encoding='utf-8') as file:
    dataset = [json.loads(line) for line in file]

# Rouge-Scores für jeden Datensatzpunkt im Datensatz berechnen
for data_point in dataset:
    candidate_summary = data_point["candidate"] #Platzhalter für LLM output
    reference_summary = data_point["summary"]

    scores = scorer.score(reference_summary, candidate_summary)

    print(f"Candidate Summary: {candidate_summary}")
    print(f"Reference Summary: {reference_summary}")

    for key in scores:
        print(f'{key}: {scores[key]}')

    print("\n")
