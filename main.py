from create_results import create_results
from metrics.bleu import run_bleu_test_on_json_dataset
from nlp.de_en import run_language_percentage
from nlp.upper_lower_case import run_upper_lower_case
from nlp.conjugated_verb import run_conjugated_verb
from metrics.perplexity_transformersGPT2 import run_perplexity_test
from metrics.rouge import run_rouge


def main():
    # create the results folder with timestamp
    output_folder = create_results()

    #Bleu
    #run_bleu_test_on_json_dataset()

    # Percentage of german & english words
    run_language_percentage(output_folder)

    # Upper lower case
    run_upper_lower_case(output_folder)

    # Conjugated Verb
    run_conjugated_verb(output_folder)

    # Perplexity
    run_perplexity_test(output_folder)

    # Rouge
    run_rouge(output_folder)


if __name__ == "__main__":
    main()

