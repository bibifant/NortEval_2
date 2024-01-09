import nltk
from nltk import word_tokenize, ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.corpus import brown
from script.azure_openai_connection import get_answer
import json
from math import exp

# Funktion zur Berechnung der Perplexität
def calculate_perplexity(model, text):
    text_ngrams = list(ngrams(word_tokenize(text), 2))
    return exp(-sum(model.logscore(ngram[-1], [ngram[0]]) if len(ngram) > 1 else model.logscore(ngram[0]) for ngram in text_ngrams) / len(text_ngrams))

# Laden des vortrainierten NLTK Bigramm-Modells
nltk.download('brown')
nltk.download('punkt')
train_data = brown.sents()
model = MLE(2)
train, vocab = padded_everygram_pipeline(2, train_data)
model.fit(train, vocab)

# Pfad JSON-Datei
json_file_path = "dataset/test.json"

# Laden des Datensets
with open(json_file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Verarbeiten der ersten zwei Zeilen
for index, line in enumerate(lines[:2]):
    # Laden des JSON-Objekts aus der Zeile
    data_point = json.loads(line)

    # Annahme: Verwendung von 'wiki_sentences' für den Prompt
    if 'wiki_sentences' in data_point:
        prompt = f"Erzähle mir mehr über: {' '.join(data_point['wiki_sentences'])}"
        response_text = get_answer(prompt, max_response_tokens=200)
        print(response_text)

        # Perplexität berechnen
        perplexity = calculate_perplexity(model, response_text)
        print(f"Perplexity: {perplexity}\n")



