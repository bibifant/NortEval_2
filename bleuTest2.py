from nltk.translate.bleu_score import sentence_bleu

# Definiere die Vorhersagen und Referenzen
predictions = ["My is age thirty years six.",
               "I am thirty six years old.",
               "I have thirty six years."]
references = [
    ["I am thirty six years old."],
    ["I'm thirty six years old."],
    ["My age is thirty six years."]
]

# Berechne die BLEU-Metrik für jede Vorhersage
for i, prediction in enumerate(predictions):
    reference_set = references[i]
    bleu_score = sentence_bleu(reference_set, prediction)
    print(f"BLEU Score for Prediction {i + 1}: {bleu_score}")
