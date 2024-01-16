import json
import os.path
from rouge_score import rouge_scorer
from script.azure_openai_connection import get_answer

def run_rouge(output_folder):
    # Rouge-Scorer initialisieren
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Pfad zur JSON-Datei
    json_file_path = "./dataset/test.json"

    # Dateipfad für die Ausgabedatei
    output_file_path = os.path.join(output_folder, "rouge_results.json")

    # Liste für Datensatzpunkte
    dataset_points = []

    # Laden der Daten aus JSON-Datei
    with open(json_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Rouge-Scores für jeden Datensatzpunkt im Datensatz berechnen
    for index, line in enumerate(lines[:2]):
        data_point = json.loads(line)

        # Attribut "wiki_sentences" zum Prompt hinzufügen
        prompt = f"Fasse den Text in deutsch zusammen:\n{data_point.get('wiki_sentences')}"

        # Zusammenfassung vom LLM
        candidate_summary = get_answer(prompt)

        # Referenz-Zusammenfassung
        reference_summary = data_point.get("klexikon_sentences", [])

        # Konvertiere die Listen in Strings
        candidate_summary_str = " ".join(candidate_summary)
        reference_summary_str = " ".join(reference_summary)

        # Rouge-Scores berechnen
        scores = scorer.score(reference_summary_str, candidate_summary_str)

        # Datensatzpunkt speichern
        dataset_point = {
            "index": index,
            "prompt": prompt,
            "candidate_summary": candidate_summary,
            "rouge1_scores": scores['rouge1'],
            "rouge2_scores": scores['rouge2'],
            "rougeL_scores": scores['rougeL']
        }

        dataset_points.append(dataset_point)

    # Rouge-Scores für Durchschnitt berechnen und speichern
    rouge1_scores = [point["rouge1_scores"].fmeasure for point in dataset_points]
    rouge2_scores = [point["rouge2_scores"].fmeasure for point in dataset_points]
    rougeL_scores = [point["rougeL_scores"].fmeasure for point in dataset_points]

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    # Durchschnitts-Rouge-Scores speichern
    avg_json_data = {
        "avg_rouge1": avg_rouge1,
        "avg_rouge2": avg_rouge2,
        "avg_rougeL": avg_rougeL
    }

    # Daten in JSON-Dateien schreiben
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(dataset_points, output_file, ensure_ascii=False, indent=2)

    # Laden der bestehenden Ergebnisdatei
    with open(os.path.join(output_folder, "avg_results.json"), 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Hinzufügen der Durchschnitts-Rouge-Scores
    existing_data["Results"].append(avg_json_data)

    # Aktualisieren der Ergebnisdatei
    with open(os.path.join(output_folder, "avg_results.json"), 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)

    return dataset_points, avg_json_data
