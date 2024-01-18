from create_results import create_results
from metrics.bleu import calculate_bleu
from metrics.perplexity_transformersGPT2 import run_perplexity_test
from metrics.rouge import run_rouge
from nlp.de_en import run_language_percentage
from nlp.gebeugtes_verb import run_gebeugtes_verb_test
from nlp.upper_lower_case import upper_lower_case


def main():
    # create the results folder with timestamp
    output_folder = create_results()

    # Tests ausführen
    # Bleu
    calculate_bleu(output_folder)

    # Funktion deutsch-englisch
    run_language_percentage(output_folder)

    # Funktion Groß- und Kleinschreibung
    upper_lower_case(output_folder)

    # Gebeugtes Verb
    # run_gebeugtes_verb_test()

    # Perplexity
    run_perplexity_test(output_folder)

    # Rouge
    run_rouge(output_folder)


if __name__ == "__main__":
    main()
