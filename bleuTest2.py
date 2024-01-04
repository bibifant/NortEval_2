from nltk.translate.bleu_score import sentence_bleu


def run_bleu_test_2(predictions, references):
    for i, prediction in enumerate(predictions):
        reference_set = references[i]
        bleu_score = sentence_bleu(reference_set, prediction)
        print(f"BLEU Score for Prediction {i + 1}: {bleu_score}")

