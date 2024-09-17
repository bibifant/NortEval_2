from huggingface_hub import HfApi

import Hugginface_Model_Scanner
from create_results import create_results_for_model
from test_classes.loading_cards_test import get_model_popularity


def select_model(models_with_popularity):
    print("Gefundene Modelle:")
    for i, (model_name, downloads, likes) in enumerate(models_with_popularity, start=1):
        print(f"{i}. Model: {model_name}, Downloads: {downloads}, Likes: {likes}")

    choice = input("Bitte wählen Sie ein Modell aus (geben Sie die Nummer ein): ")
    try:
        index = int(choice) - 1
        if 0 <= index < len(models_with_popularity):
            return models_with_popularity[index][0]  # Gib den ausgewählten Modellnamen zurück
        else:
            print("Ungültige Auswahl. Bitte geben Sie eine gültige Nummer ein.")
            return None
    except ValueError:
        print("Ungültige Eingabe. Bitte geben Sie eine Zahl ein.")
        return None


def get_model_details(model_name):
    api = HfApi()
    model_info = api.model_info(model_name)
    return model_info


def main():
    model_names = Hugginface_Model_Scanner.generate_list_of_available_german_models()

    # Erstelle eine Liste mit Modellnamen und deren Popularität
    models_with_popularity = []
    for model_name in model_names:
        downloads, likes = get_model_popularity(model_name)
        models_with_popularity.append((model_name, downloads, likes))

    # Sortiere die Modelle nach der Anzahl der Downloads (absteigend)
    models_with_popularity.sort(key=lambda x: x[1], reverse=True)

    # Benutzer wählt ein Modell aus
    selected_model = select_model(models_with_popularity)
    if selected_model:
        # Lade Modellinformationen und führe Benchmark aus
        model_info = get_model_details(selected_model)
        print(f"Ausgewähltes Modell: {selected_model}")
        output_folder = create_results_for_model(selected_model)

    else:
        print("Kein gültiges Modell ausgewählt. Benchmark wird nicht ausgeführt.")


if __name__ == "__main__":
    main()
