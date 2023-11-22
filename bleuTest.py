import evaluate

bleu = evaluate.load("bleu")
predictions = ["I have thirty six years. I also have great hair and I run really fast."]
references = [
    ["I am thirty six years old. I do have great hair an I'm the fastest runner in town."]
]
results = bleu.compute(predictions=predictions, references=references)
print(results)