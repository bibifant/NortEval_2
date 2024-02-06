from create_results import create_results
from metrics.bleu import calculate_bleu
from metrics.rouge import run_rouge
from nlp.bias_detection import calculate_bias
from nlp.contains_verb import run_contains_verb
from nlp.de_en import run_language_percentage
from nlp.sentiment_analysis import run_sentiment_analysis
from nlp.upper_lower_case import run_upper_lower_case
from nlp.natural_language_quality_tests.natural_language_quality_assessor import evaluate_generated_text_quality


def main():
    # Create The Results Folder With Timestamp
    output_folder = create_results()

    # Bias
    calculate_bias(output_folder)

    # Sentiment Analysis
    run_sentiment_analysis(output_folder)

    # Bleu
    calculate_bleu(output_folder)

    # Natural Language/ Plausibility
    evaluate_generated_text_quality(output_folder)

    # Deutsch-Englisch
    run_language_percentage(output_folder)

    # Rouge
    run_rouge(output_folder)

    # Percentage of German & English words
    run_language_percentage(output_folder)

    # Upper Lower Case
    run_upper_lower_case(output_folder)

    # Conjugated Verb
    run_contains_verb(output_folder)


if __name__ == "__main__":
    main()
