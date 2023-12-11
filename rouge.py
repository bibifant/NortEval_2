import json
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Pfad JSONL-Datei
json_file_path = "test.json"

# Daten aus JSONL-Datei lesen
with open(json_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Rouge-Scores für jeden Datensatzpunkt im Datensatz berechnen
for line in lines[:2]:
    data_point = json.loads(line)
    candidate_summary = data_point.get("wiki_sentences", [])  # Platzhalter für LLM output
    reference_summary = data_point.get("klexikon_sentences", [])

    # Konvertiere die Listen in Strings
    candidate_summary_str = " ".join(candidate_summary)
    reference_summary_str = " ".join(reference_summary)

    scores = scorer.score(reference_summary_str, candidate_summary_str)

    print(f"Candidate Summary: {candidate_summary_str}")
    print(f"Reference Summary: {reference_summary_str}")

    for key in scores:
        print(f'{key}: {scores[key]}')

    print("\n")