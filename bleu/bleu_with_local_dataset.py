import json
from script.azure_openai_connection import get_simple_translation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# link to dataset: http://linguatools.org/webcrawl-parallel-corpus-german-english-2015/
# input file paths
de_file_path = "dataset/zitate-dewiki-20141024.de"
en_file_path = "dataset/zitate-dewiki-20141024.en"
# create output file path
output_file_path = "results/bleu_with_local_dataset_results.json"
"""
This file performs a BLEU test on long text.
The BLEU score is calculated in the following steps:
1. Retrieve the German dataset from 'dataset/'.
2. Retrieve the English dataset from 'dataset/'.
3. The 'de_content' text is used as the human reference, and the 'en_content' text is used as predictions, which need to be translated into German.
4. OpenAI is used to translate the text of 'en_content' into German, and 'sentence_bleu()' is used to calculate the BLEU score.
5. Check for errors; only non-None predictions and references will be calculated.
6. Reformat the score to display it as a percentage (e.g., 0.80).
7. Calculate the average score.
8. Save all the results as a JSON file.
"""


def calculate_bleu(max_index=100):
    # Read content from the ".de" file
    with open(de_file_path, 'r', encoding='utf-8') as de_file:
        de_content = de_file.read()

    # Read content from the ".en" file
    with open(en_file_path, 'r', encoding='utf-8') as en_file:
        en_content = en_file.read()

    # Split the content into sentences (assuming each line is a sentence)
    de_sentences = de_content.split('\n')
    en_sentences = en_content.split('\n')

    # Initialise total score
    total_bleu_score = 0
    results = {"scores": []}
    count = 0  # Number of successful BLEU score calculations

    for i, (en_sentence, de_sentence) in enumerate(zip(en_sentences, de_sentences)):
        prediction = get_simple_translation(en_sentence)  # Translate 'en' prediction to German using OpenAI
        reference = de_sentence  # 'de' is used as a reference

        # Check if prediction or reference is none
        if not prediction or not reference:
            print(f"Incorrect datapoint {i + 1}: Missing human reference or prediction.")
        else:
            # Calculation of the BLEU score for the prediction compared to the human reference
            bleu_score = sentence_bleu([prediction.split()], reference.split(),
                                       smoothing_function=SmoothingFunction().method1,
                                       weights=(1, 0))  # Calculate BLEU with unigram
            total_bleu_score += bleu_score  # Set total BLEU score

            # Formatting the BLEU score to two decimal places after the decimal point
            formatted_bleu_score = "{:.2f}".format(bleu_score)

            # Save the results as a dictionary
            results["scores"].append({
                "prediction_index": i + 1,
                "bleu_score": formatted_bleu_score
            })
            count += 1

        # Stop loop if it has iterated `max_index` times
        if i >= max_index:
            break

    # Average BLEU score for the entire dataset
    average_bleu_score = total_bleu_score / count
    # Formatting the average BLEU score to two decimal places after the decimal point
    formatted_bleu_average_score = "{:.2f}".format(average_bleu_score)
    results["count"] = count
    results["average_bleu_score"] = formatted_bleu_average_score

    # Saving the results as JSON
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=2)
