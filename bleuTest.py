import evaluate


def run_bleu_test_1(predictions, references):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    print(results)
