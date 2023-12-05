import evaluate

# Lade die BLEU-Metrik
bleu = evaluate.load("bleu")
predictions = ["I have thirty six years old."]
references = ["I'm thirty six years old."]
results = bleu.compute(predictions=predictions, references=references)
print(results)