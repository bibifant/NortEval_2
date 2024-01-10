from create_results_file import create_results_file
from bleu import run_bleu_test_on_json_dataset
from de_en import run_de_en_test
from functionGK import run_function_gk_test
from gebeugtes_verb import run_gebeugtes_verb_test
from perplexity_transformersGPT2 import run_perplexity_test
from rouge import run_rouge


def main():
    # Erstelle zuerst die results.json Datei
    output_file = create_results_file()

    # Setze die Testdaten entsprechend
    json_file_bleu3_path = "dataset/bleu_dataset.json"

    example_text_de_en = "Dies ist an english example text und ein deutscher Text zum Überprüfen."

    text_example_gk = "Der schnelle braune Fuchs springt über den faulen Hund und der Fuchs freut sich. Der große Baum."

    sentences_gebeugtes_verb = ["Der Hund spielt im Park.",
                                "Die Vögel singen fröhlich.",
                                "lachen, spielen, singen.",
                                "Fenster singen im Park."]

    # Führe die Tests aus
    print("\n\nBLEU:")
    run_bleu_test_on_json_dataset(json_file_bleu3_path)

    print("\n\nde_en:")
    run_de_en_test(example_text_de_en)

    print("\n\nFunction GK:")
    run_function_gk_test(text_example_gk)

    print("\n\nGebeugtes Verb:")
    run_gebeugtes_verb_test(sentences_gebeugtes_verb)

    print("\n\nPerplexity:")
    run_perplexity_test(output_file)

    print("\n\nRouge:")
    run_rouge()


if __name__ == "__main__":
    main()

