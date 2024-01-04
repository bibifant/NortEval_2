import json
from rouge_score import rouge_scorer
import sys
sys.path.append('script')
from script.azure_openai_connection import get_answer


def rouge():
    # Initialisiere den ROUGE-Scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Pfad JSON-Datei
    json_file_path = "test.json"

    # Pfad für die Ausgabedatei
    output_file_path = "llm_evaluation_rouge.txt"

    # Öffne die Ausgabedatei im Schreibmodus
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Lade die Daten aus der JSON-Datei
        with open(json_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Rouge-Scores für jeden Datensatzpunkt im Datensatz berechnen
        for index, line in enumerate(lines[:2]):
            data_point = json.loads(line)

            # Füge das Attribut "wiki_sentences" zum Prompt hinzu
            prompt = f"Fasse den Text in 10 Sätzen in deutsch zusammen:\n{data_point.get('wiki_sentences')}"

            # Schreibe den Prompt mit Index in die Datei
            output_file.write(f"Index: {index}\n")
            output_file.write(f"Prompt: {prompt}\n")

            # Erhalte die Zusammenfassung vom LLM
            candidate_summary = get_answer(prompt)

            # Schreibe den LLM-Output in die Datei
            output_file.write(f"LLM Output: {candidate_summary}\n")

            # Erhalte die Referenz-Zusammenfassung
            reference_summary = data_point.get("klexikon_sentences", [])

            # Konvertiere die Listen in Strings
            candidate_summary_str = " ".join(candidate_summary)
            reference_summary_str = " ".join(reference_summary)

            # Berechne ROUGE-Scores
            scores = scorer.score(reference_summary_str, candidate_summary_str)

            # Schreibe die ROUGE-Scores in die Datei
            output_file.write("ROUGE Scores:\n")
            for key in scores:
                output_file.write(f'{key}: {scores[key]}\n')

            output_file.write("\n")


# Aufruf der Funktion
rouge()