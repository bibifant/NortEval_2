from bleuTest import run_bleu_test_1
from bleuTest2 import run_bleu_test_2
from de_en import run_de_en_test
from functionGK import run_function_gk_test
from gebeugtes_verb import run_gebeugtes_verb_test
from perplexityTest import run_perplexity_test
# from rouge import run_rouge_test


def main():
    # Setze die Testdaten entsprechend
    predictions_bleu1 = ["I have thirty six years old."]
    references_bleu1 = ["I'm thirty six years old."]

    predictions_bleu2 = ["My is age thirty years six.",
                         "I am thirty six years old.",
                         "I have thirty six years."]
    references_bleu2 = [
        ["I am thirty six years old."],
        ["I'm thirty six years old."],
        ["My age is thirty six years."]
    ]

    example_text_de_en = "Dies ist an english example text und ein deutscher Text zum Überprüfen."

    text_example_gk = "Der schnelle braune Fuchs springt über den faulen Hund und der Fuchs freut sich. Der große Baum."

    sentences_gebeugtes_verb = ["Der Hund spielt im Park.",
                                "Die Vögel singen fröhlich.",
                                "lachen, spielen, singen.",
                                "Fenster singen im Park."]

    dataset_path_perplexity = "WikiQA-train.txt"

    jsonl_file_path_rouge = "de_test.jsonl" #dataset muss updated werden

    # Führe die Tests aus
    print("\n\nBLEU Test 1:")
    run_bleu_test_1(predictions_bleu1, references_bleu1)

    print("\n\nBLEU Test 2:")
    run_bleu_test_2(predictions_bleu2, references_bleu2)

    print("\n\nde_en Test:")
    run_de_en_test(example_text_de_en)

    print("\n\nFunction GK Test:")
    run_function_gk_test(text_example_gk)

    print("\n\nGebeugtes Verb Test:")
    run_gebeugtes_verb_test(sentences_gebeugtes_verb)

    print("\n\nPerplexity Test:")
    run_perplexity_test(dataset_path_perplexity)
    #
    # print("\n\nRouge Test:")
    # run_rouge_test(jsonl_file_path_rouge)


if __name__ == "__main__":
    main()

