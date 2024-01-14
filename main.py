from create_results import create_results
from metrics.bleu import run_bleu_test_on_json_dataset
from nlp.de_en import run_de_en_test
from nlp.functionGK import run_function_gk_test
from nlp.gebeugtes_verb import run_gebeugtes_verb_test
from metrics.perplexity_transformersGPT2 import run_perplexity_test
from metrics.rouge import run_rouge


def main():
    #Erstelle zuerst den results.json Ordner mit Zeitstempel
    output_folder = create_results()

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
    run_perplexity_test(output_folder)

    #Rouge
    run_rouge(output_folder)


if __name__ == "__main__":
    main()
