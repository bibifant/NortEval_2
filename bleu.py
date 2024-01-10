import json
from nltk.translate.bleu_score import sentence_bleu


def run_bleu_test_on_json_dataset(json_file_path, output_file_path):
    # Öffnen der JSON-Datei
    with open(json_file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # Initialisierung des Gesamtscores
    total_bleu_score = 0

    # Öffnen der Ausgabedatei im Schreibmodus
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Kurze Erklärung zu BLEU und BLEU-Score am Anfang der Datei
        explanation = (
            "BLEU (Bilingual Evaluation Understudy) ist eine Metrik zur Bewertung der Qualität von maschinellen Übersetzungen. "
            "Der BLEU-Score misst die Ähnlichkeit zwischen einer automatisch generierten Übersetzung und einer oder mehreren Referenzübersetzungen. "
            "Ein höherer BLEU-Score deutet darauf hin, dass die automatische Übersetzung besser mit den Referenzübersetzungen übereinstimmt."
        )

        # Schreibe Erklärung in die Datei
        output_file.write(explanation + '\n\n')
        for i, data_point in enumerate(dataset):
            # Jeder Datenpunkt enthält "predictions" und "references"
            predictions = data_point.get("predictions", [])
            references = data_point.get("references", [])

            # Überprüfen, ob der Datenpunkt Vorhersagen und Referenzen enthält
            if not predictions or not references:
                output_file.write(f"Fehlerhafter Datenpunkt {i + 1}: Fehlende Vorhersagen oder Referenzen.\n")
                continue

            for j, prediction in enumerate(predictions):
                # Abbruch, wenn mehr Vorhersagen als Referenzen vorhanden sind
                if j >= len(references):
                    break

                reference_set = references[j]
                # Berechnung des BLEU-Scores für die Vorhersage
                bleu_score = sentence_bleu(reference_set, prediction)
                total_bleu_score += bleu_score

                output_file.write(f"BLEU Score für Vorhersage {i + 1}-{j + 1}: {bleu_score}\n")

    # Durchschnittlicher BLEU-Score für das gesamte Dataset
    average_bleu_score = total_bleu_score / len(dataset)
    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        output_file.write(
            f"\nDurchschnittlicher BLEU-Score für das gesamte Dataset({len(dataset)}): {average_bleu_score}\n")
