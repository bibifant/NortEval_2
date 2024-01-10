import json
from datetime import datetime

def create_results_file():
    # Aktuellen Zeitstempel im Format "YYYYMMDD-HHMMSS" erhalten
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"results_{timestamp}.json"

    data_structure = {
        "Metrik Erklärungen": {
            "Rouge": "Platzhalter für Erklärung der Rouge Metrik",
            "BLEU": "Platzhalter für Erklärung der BLEU Metrik",
            "Perplexity": "Perplexity ist ein Maß für die Vorhersageunsicherheit eines Sprachmodells. Ein niedrigerer Wert deutet auf eine höhere Vorhersagegenauigkeit des Modells hin."
        },
        "Ergebnisse": []
    }

    with open(file_name, mode='w', encoding='utf-8') as file:
        json.dump(data_structure, file, ensure_ascii=False, indent=4)
    print(f"{file_name} wurde erstellt.")

    return file_name
