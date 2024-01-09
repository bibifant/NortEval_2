import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from math import exp
import json
from script.azure_openai_connection import get_answer

# Funktion zur Berechnung der Perplexität
def calculate_perplexity(model, tokenizer, text):
    tokenize_input = tokenizer.tokenize(text)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, labels=tensor_input).loss
    return exp(loss.item())

# Laden des Modells und des Tokenizers
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Liste von Prompts
prompts = [
    "Diskutiere die ethischen Implikationen der künstlichen Intelligenz.",  # Ethische Frage
    "Erkläre die Funktionsweise eines Schwarzen Lochs.",  # Faktische Informationen
    "Schreibe eine Kurzgeschichte über eine Zeitreise ins Mittelalter.",  # Kreative Schreibweise
    "Erläutere den Unterschied zwischen Maschinellem Lernen und Tiefem Lernen.",  # Technische Erklärung
    "Wie plant man eine effektive Wochenmahlzeit für eine vierköpfige Familie?", # Alltägliches
    "Beschreibe die Entwicklung der klassischen Musik im 20. Jahrhundert.", # Kulturelle Themen
    "Wie geht man mit Arbeitsstress um?",  # Emotionale Beratung
    "Wie könnte die Welt im Jahr 2050 aussehen, wenn erneuerbare Energien die Hauptenergiequelle sind?"  # Hypothetische Szenarien
    ]

# Ergebnisliste für aktuelle Session
current_results = []
total_perplexity = 0

for prompt in prompts:
    try:
        # Antwort erhalten
        response_text = get_answer(prompt)

        # Perplexität berechnen
        perplexity = calculate_perplexity(model, tokenizer, response_text)
        total_perplexity += perplexity

        # Ergebnisse in der aktuellen Liste speichern
        current_results.append({
            "Prompt": prompt,
            "Response": response_text,
            "Perplexity Score": perplexity
        })
    except Exception as e:
        print(f"Fehler bei der Verarbeitung des Prompts '{prompt}': {e}")

# Durchschnittsperplexität berechnen
average_perplexity = total_perplexity / len(prompts) if prompts else 0

# Dateiname für die Ergebnisse
output_file = "results.json"

# Vorhandene JSON-Datei laden
with open(output_file, 'r', encoding='utf-8') as file:
    data_structure = json.load(file)

# Aktuelle Ergebnisse zu den bestehenden hinzufügen
data_structure["Ergebnisse"].extend(current_results)

# Durchschnittliche Perplexität am Ende der Ergebnisse hinzufügen
data_structure["Ergebnisse"].append({"Durchschnittliche Perplexität": average_perplexity})


# JSON-Datei mit aktualisierten Daten schreiben
with open(output_file, mode='w', encoding='utf-8') as file:
    json.dump(data_structure, file, ensure_ascii=False, indent=4)

print(f"Ergebnisse und Durchschnittliche Perplexität wurden in {output_file} aktualisiert.")