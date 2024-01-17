import json
import os.path
from datetime import datetime

def create_results():
    # Receive the current timestamp in the format "YYYYMMDD-HHMMSS"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"results_{timestamp}"

    # Check whether the folder exists. If not, create it.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create file name for the average result file
    file_name = os.path.join(folder_name, "avg_results.json")

    data_structure = {
        "Metrik Erklärungen": {
            "Rouge": "Die Rouge-Metrik bewertet die Fähigkeit eines LLMs, Zusammenfassungen aus einem Input Text zu generieren. Der Rouge-Score reicht von 0 bis 1, wobei höhere Werte eine bessere Qualität der Zusammenfassung anzeigen.",
            "BLEU": "BLEU ist eine Metrik zur Bewertung der Qualität von maschinellen Übersetzungen. Bleu Score reicht von 0 bis 1. Ein höherer BLEU-Score deutet darauf hin, dass die automatische Übersetzung besser mit den Referenzübersetzungen übereinstimmt.",
            "Perplexity": "Perplexity ist ein Maß für die Vorhersageunsicherheit eines Sprachmodells. Ein niedrigerer Wert deutet auf eine höhere Vorhersagegenauigkeit des Modells hin. Ein Wert nahe 50 ist hervorragend."
        },
        "Results": []
    }

    with open(file_name, mode='w', encoding='utf-8') as file:
        json.dump(data_structure, file, ensure_ascii=False, indent=4)
    print(f"The folder {folder_name} has been created.")

    return folder_name
