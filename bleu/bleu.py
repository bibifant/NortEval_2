import json
from bleu.utils import fetch_dataset_from_api
from bleu.utils import extract_translations
from script.azure_openai_connection import get_simple_translation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(output_file_path):
    # Öffnen der JSON-Datei
    data = fetch_dataset_from_api()
    dataset = extract_translations(data)
    print(dataset)
    # Initialisierung des Gesamtscore
    total_bleu_score = 0
    results = {"scores": []}

    for i, data_point in enumerate(dataset):
        # Jeder Datenpunkt enthält "prediction" und "referenz"
        # übersetzt nach Deutsche Sprache
        print(f"original_text: {data_point['en']}")
        # übersetzt englische prediction zu Deutsch. Danach kan mit referenz berechnen
        prediction = get_simple_translation(data_point.get("en", ""))
        print(f"translated_text: {prediction}")
        referenz = data_point.get("de", "")

        # Überprüfen, ob der Datenpunkt menschliche Referenz und Vorhersage enthält
        if not prediction or not referenz:
            print(f"Fehlerhafter Datenpunkt {i + 1}: Fehlende menschliche Referenz oder vorhersage.")
            continue

        # Berechnung des BLEU-Scores für die Vorhersage im Vergleich zur menschlichen Referenz
        bleu_score = sentence_bleu([prediction.split()], referenz.split(),
                                   smoothing_function=SmoothingFunction().method1,
                                   weights=(1, 0))  # bleu nach uni-gramm
        total_bleu_score += bleu_score

        # Formatieren des BLEU-Scores auf zwei Dezimalstellen nach dem Komma
        formatted_bleu_score = "{:.2f}".format(bleu_score)

        # Speichern der Ergebnisse als Dictionary
        results["scores"].append({
            "prediction_index": i + 1,
            "bleu_score": formatted_bleu_score
        })

    # Durchschnittlicher BLEU-Score für das gesamte Dataset
    average_bleu_score = total_bleu_score / len(dataset)
    # Formatieren des BLEU-Scores auf zwei Dezimalstellen nach dem Komma
    formatted_bleu_average_score = "{:.2f}".format(average_bleu_score)
    results["average_bleu_score"] = formatted_bleu_average_score

    # Speichern der Ergebnisse als JSON
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=2)
