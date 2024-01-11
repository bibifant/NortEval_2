from create_results import create_results
from metrics.bleu import run_bleu_test_on_json_dataset
from nlp.de_en import run_de_en_test
from nlp.functionGK import run_function_gk_test
from nlp.gebeugtes_verb import run_gebeugtes_verb_test
from metrics.perplexity_transformersGPT2 import run_perplexity_test
from metrics.rouge import run_rouge


def main():
    #Erstelle zuerst die results.json Datei
    output_file = create_results()

    #Tests ausführen
    #Bleu
    #run_bleu_test_on_json_dataset()

    #Funktion deutsch-englisch
    #run_de_en_test()

    #Funktion Groß- und Kleinschreibung
    #run_function_gk_test()

    #Gebeugtes Verb
    #run_gebeugtes_verb_test()

    #Perplexity
    run_perplexity_test(output_file)

    #Rouge
    run_rouge(output_file)


if __name__ == "__main__":
    main()
