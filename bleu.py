# Diese Funktion führt einen BLEU-Test auf einem JSON-Dataset durch.

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def run_bleu_test_on_json_dataset(json_file_path):
    # Öffnen der JSON-Datei
    with open(json_file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # Initialisierung des Gesamtscores
    total_bleu_score = 0

    # SmoothingFunction wird verwendet, um Warnmeldungen zu vermeiden
    smoothing = SmoothingFunction()

    for i, data_point in enumerate(dataset):
        # Jeder Datenpunkt enthält "predictions" und "references"
        predictions = data_point.get("predictions", [])
        references = data_point.get("references", [])

        # Überprüfen, ob der Datenpunkt Vorhersagen und Referenzen enthält
        if not predictions or not references:
            print(f"Fehlerhafter Datenpunkt {i + 1}: Fehlende Vorhersagen oder Referenzen.")
            continue

        for j, prediction in enumerate(predictions):
            # Abbruch, wenn mehr Vorhersagen als Referenzen vorhanden sind
            if j >= len(references):
                break

            reference_set = references[j]
            print(f"Referenz-Set: {reference_set}")
            print(f"Vorhersage: {prediction}")

            # Berechnung des BLEU-Scores für die Vorhersage
            bleu_score = sentence_bleu(reference_set, prediction, smoothing_function=smoothing.method1)
            total_bleu_score += bleu_score

            print(f"BLEU Score für Vorhersage {i + 1}-{j + 1}: {bleu_score}")

    # Durchschnittlicher BLEU-Score für das gesamte Dataset
    average_bleu_score = total_bleu_score / len(dataset)
    print(f"\nDurchschnittlicher BLEU-Score für das gesamte Dataset: {average_bleu_score}")


# Beispielaufruf:
# json_file_path = "dataset/bleu_dataset.json"
# run_bleu_test_on_json_dataset(json_file_path)
