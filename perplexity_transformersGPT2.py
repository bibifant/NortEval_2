import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from math import exp
import csv
import os
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

# Dateiname für die Ergebnisse
output_file = "perplexity_results.csv"

# Überprüfen, ob die Datei bereits existiert und ob eine Kopfzeile erforderlich ist
write_header = not os.path.exists(output_file)

# CSV-Datei öffnen und Ergebnisse schreiben
with open(output_file, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Schreibt Kopfzeile nur, wenn die Datei neu ist
    if write_header:
        writer.writerow(['Prompt', 'Response', 'Perplexity'])

    for prompt in prompts:
        try:
            # Antwort erhalten
            response_text = get_answer(prompt)

            # Perplexität berechnen
            perplexity = calculate_perplexity(model, tokenizer, response_text)

            # Ergebnisse in CSV schreiben
            writer.writerow([prompt, response_text, perplexity])
        except Exception as e:
            print(f"Fehler bei der Verarbeitung des Prompts '{prompt}': {e}")

print(f"Ergebnisse wurden in {output_file} gespeichert.")


