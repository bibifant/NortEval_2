import json
from rouge_score import rouge_scorer
import sys
sys.path.append('script')
from script.azure_openai_connection import get_answer


def run_rouge():
    # rouge score initialisieren
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Pfad JSON-Datei
    json_file_path = "dataset/test.json"

    # Pfad für die Ausgabedatei
    output_file_path = "llm_evaluation_rouge.txt"

    # Pfad für die Ausgabedatei Durchschnitt
    output_average_file_path = "llm_evaluation_all.txt"

    # Listen für rouge scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Öffnen der Ausgabedatei im Schreibmodus
    with open(output_file_path, 'w', encoding='utf-8') as output_file:

        # Laden der Daten aus JSON-Datei
        with open(json_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        output_file.write(f"Der Rouge-Score reicht von 0 bis 1, wobei höhere Werte eine bessere Qualität der Zusammenfassung anzeigen.\n\n")

        # rouge scores für jeden Datensatzpunkt im Datensatz berechnen
        for index, line in enumerate(lines[:2]):
            data_point = json.loads(line)

            # Attribut "wiki_sentences" zum Prompt hinzufügen
            prompt = f"Fasse den Text in 10 Sätzen in deutsch zusammen:\n{data_point.get('wiki_sentences')}"

            # Prompt und Index in Datei schreiben
            output_file.write(f"Index: {index}\n")
            output_file.write(f"Prompt: {prompt}\n")

            # Zusammenfassung vom LLM
            candidate_summary = get_answer(prompt)

            # LLM-Output in Datei schreiben
            output_file.write(f"LLM Output: {candidate_summary}\n")

            # Referenz-Zusammenfassung
            reference_summary = data_point.get("klexikon_sentences", [])

            # Konvertiere die Listen in Strings
            candidate_summary_str = " ".join(candidate_summary)
            reference_summary_str = " ".join(reference_summary)

            # rouge scores berechnen
            scores = scorer.score(reference_summary_str, candidate_summary_str)

            # rouge scores speichern
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # rouge scores in Datei schreiben
            output_file.write("Rouge Scores:\n")
            for key in scores:
                output_file.write(f'{key}: {scores[key]}\n')

            output_file.write("\n")

    # Durchschnitt berechnen
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    # Durchschnitt in Datei schreiben
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write("\nDurchschnittliche Rouge Scores:\n")
        output_file.write(f'rouge1: {avg_rouge1}\n')
        output_file.write(f'rouge2: {avg_rouge2}\n')
        output_file.write(f'rougeL: {avg_rougeL}\n')

    # Durchschnitt in Durchschnitt-Datei schreiben
    with open(output_average_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write("\nDurchschnittliche Rouge Scores:\n")
        output_file.write(f'rouge1: {avg_rouge1}\n')
        output_file.write(f'rouge2: {avg_rouge2}\n')
        output_file.write(f'rougeL: {avg_rougeL}\n')


