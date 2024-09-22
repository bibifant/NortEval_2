import os
import json
from huggingface_hub import HfApi
import Hugginface_Model_Scanner
from create_results import create_results_for_model
from metrics.bleu import calculate_bleu
from metrics.rouge import run_rouge
from nlp.bias_detection.hate_speech_detection import run_hate_speech
from nlp.bias_detection.word_connotation_recognition import run_word_connotation_recognition
from nlp.sentiment_analysis_user_comments import run_sentiment_analysis
from openai_connection import load_model_and_tokenizer
from test_classes.loading_cards_test import get_model_popularity
import main as benchmark_tests


def select_model(models_with_popularity):
    print("Gefundene Modelle:")
    for i, (model_name, downloads, likes) in enumerate(models_with_popularity, start=1):
        print(f"{i}. Model: {model_name}, Downloads: {downloads}, Likes: {likes}")

    choice = input("Bitte wählen Sie ein Modell aus (geben Sie die Nummer ein): ")
    try:
        index = int(choice) - 1
        if 0 <= index < len(models_with_popularity):
            return models_with_popularity[index][0]  # Gibt den ausgewählten Modellnamen zurück
        else:
            print("Ungültige Auswahl. Bitte geben Sie eine gültige Nummer ein.")
            return None
    except ValueError:
        print("Ungültige Eingabe. Bitte geben Sie eine Zahl ein.")
        return None

#
# def get_model_details_manually(model_name):
#     api = HfApi()
#     model_info = api.model_info(model_name)
#     return model_info


def get_available_models():
    model_names = Hugginface_Model_Scanner.generate_list_of_available_german_models()

    # Get popularity for each model
    models_with_popularity = []
    for model_name in model_names:
        downloads, likes = get_model_popularity(model_name)
        models_with_popularity.append((model_name, downloads, likes))

    # Sort models by popularity
    models_with_popularity.sort(key=lambda x: x[1], reverse=True)

    return models_with_popularity


def main():
    model_names = Hugginface_Model_Scanner.generate_list_of_available_german_models()

    # Erstellt eine Liste mit Modellnamen und deren Popularität
    models_with_popularity = []
    for model_name in model_names:
        downloads, likes = get_model_popularity(model_name)
        models_with_popularity.append((model_name, downloads, likes))

    # Sortiert die Modelle nach der Anzahl der Downloads (absteigend)
    models_with_popularity.sort(key=lambda x: x[1], reverse=True)

    # Benutzer kann ein Modell auswählen
    selected_model = select_model(models_with_popularity)
    if selected_model:
        # Lädt Modellinformationen und führt Benchmark aus
        print(f"Ausgewähltes Modell: {selected_model}")
        benchmark_tests.main()

    else:
        print("Kein gültiges Modell ausgewählt. Benchmark wird nicht ausgeführt.")


def run_benchmark_on_model(selected_model_name):
    # Model und Tokenizer laden
    model, tokenizer = load_model_and_tokenizer(selected_model_name)

    # Sanitize the model name for the folder and file naming
    sanitized_model_name = selected_model_name.replace('/', '_')

    # Erstelle den Ordner für die Ergebnisse
    output_folder = create_results_for_model(selected_model_name)

    # Führe die Benchmark-Tests aus
    results = []

    # Benchmarktests aufrufen
    bleu_score = calculate_bleu(output_folder, model)
    results.append({"name": "BLEU", "score": bleu_score})

    rouge_score = run_rouge(model, tokenizer, output_folder)
    results.append({"name": "ROUGE", "score": rouge_score})

    hate_speech_score = run_hate_speech(model, tokenizer, output_folder)
    results.append({"name": "Hate Speech Detection", "score": hate_speech_score})
    #
    # word_connotation_score = run_word_connotation_recognition(model, tokenizer, output_folder)
    # results.append({"name": "Word Connotation Recognition", "score": word_connotation_score})
    #
    # sentiment_analysis_score = run_sentiment_analysis(model, tokenizer, output_folder)
    # results.append({"name": "Sentiment Analysis", "score": sentiment_analysis_score})

    # Hier können weitere Tests hinzugefügt werden...

    # # Ergebnisse speichern
    results_file = os.path.join(output_folder, f"{sanitized_model_name}_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({"Results": results}, f, ensure_ascii=False, indent=4)

    return {output_folder, selected_model_name}


if __name__ == "__main__":
    main()
