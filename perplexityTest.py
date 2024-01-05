import nltk
from nltk import FreqDist
from nltk.lm import MLE
from nltk.util import bigrams
import math


# 1. Datenset laden und vorverarbeiten
def load_and_preprocess_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    return raw_text


# 2. Tokenisierung mit nltk
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens


# 3. Perplexity berechnen
def calculate_perplexity(tokens, ngram_order=2):
    # NLTK MLE-Modell trainieren
    bigram_model = MLE(ngram_order)
    vocab = set(tokens)
    bigram_model.fit([bigrams(tokens)], vocabulary_text=vocab)

    # Wahrscheinlichkeiten für jedes Bigramm im Text berechnen
    probabilities = [bigram_model.score(bigram) for bigram in bigrams(tokens)]

    # Make sure probabilities are not 0, as log(0) is undefined
    probabilities = [p if p != 0 else 1e-10 for p in probabilities]

    # Perplexity berechnen
    perplexity = math.exp(-sum(math.log(p) for p in probabilities) / len(probabilities))

    return perplexity


def run_perplexity_test(dataset_path):
    print(f"perplexity: {calculate_perplexity(tokenize_text(load_and_preprocess_dataset(dataset_path)))}")

# dataset_path = "WikiQA-train.txt"
# run_perplexity_test(dataset_path)
# # Beispiel-Datenset laden und vorverarbeiten
# raw_text = load_and_preprocess_dataset(dataset_path)
#
# # Beispiel-Tokenisierung
# tokens = tokenize_text(raw_text)
#
# # Beispiel-Perplexity berechnen (mit einem NLTK MLE-Modell)
# perplexity = calculate_perplexity(tokens)
#
# print(f"Perplexity: {perplexity}")
