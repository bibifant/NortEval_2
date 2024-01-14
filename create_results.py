import json
import os.path
from datetime import datetime

def create_results():
    # Aktuellen Zeitstempel im Format "YYYYMMDD-HHMMSS" erhalten
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"results_{timestamp}"

    # Überprüfe, ob der Ordner existiert. Wenn nicht, erstelle ihn.
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Erstelle den Dateinamen für die Ergebnisdatei
    file_name = os.path.join(folder_name, "avg_results.json")

    data_structure = {
        "Metrik Erklärungen": {
            "Rouge": "Die Rouge-Metrik bewertet die Fähigkeit eines LLMs, Zusammenfassungen aus einem Input Text zu generieren. Der Rouge-Score reicht von 0 bis 1, wobei höhere Werte eine bessere Qualität der Zusammenfassung anzeigen.",
            "BLEU": "BLEU ist eine Metrik zur Bewertung der Qualität von maschinellen Übersetzungen. Bleu Score reicht von 0 bis 1. Ein höherer BLEU-Score deutet darauf hin, dass die automatische Übersetzung besser mit den Referenzübersetzungen übereinstimmt.",
            "Perplexity": "Perplexity ist ein Maß für die Vorhersageunsicherheit eines Sprachmodells. Ein niedrigerer Wert deutet auf eine höhere Vorhersagegenauigkeit des Modells hin. Ein Wert nahe 50 ist hervorragend."
        },
        "Ergebnisse": []
    }

    with open(file_name, mode='w', encoding='utf-8') as file:
        json.dump(data_structure, file, ensure_ascii=False, indent=4)
    print(f"Der Ordner {folder_name} wurde erstellt.")

    return folder_name
