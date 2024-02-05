from create_results import create_results
from metrics.bleu import calculate_bleu
from metrics.perplexity_transformersGPT2 import run_perplexity_test
from metrics.rouge import run_rouge
from nlp.bias_detection import calculate_bias
from nlp.contains_verb import run_contains_verb
from nlp.de_en import run_language_percentage
from nlp.sentiment_analysis import run_sentiment_analysis
from nlp.upper_lower_case import run_upper_lower_case


def main():
    # create the results folder with timestamp
    output_folder = create_results()

    # Bias
    calculate_bias(output_folder)

    # sentiment analysis
    run_sentiment_analysis(output_folder)

    # Bleu
    calculate_bleu(output_folder)

    # Deutsch-englisch
    run_language_percentage(output_folder)

    # Perplexity
    run_perplexity_test(output_folder)

    # Rouge
    run_rouge(output_folder)

    # Percentage of german & english words
    run_language_percentage(output_folder)

    # Upper lower case
    run_upper_lower_case(output_folder)

    # Conjugated Verb
    run_contains_verb(output_folder)


if __name__ == "__main__":
    main()
