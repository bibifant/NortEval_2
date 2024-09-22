import json
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Import Hugging Face translation functions from the modified openai_connection
from openai_connection import load_model_and_tokenizer, get_simple_translation

# File paths for datasets
de_file_path = "datasets/zitate-dewiki-20141024.de"
en_file_path = "datasets/zitate-dewiki-20141024.en"


def categorize_bleu_score(bleu_score):
    if bleu_score > 0.4:
        return "good"
    elif 0.2 <= bleu_score <= 0.4:
        return "average"
    else:
        return "bad"


def calculate_bleu(model_name, max_index=10):
    print(f"BLEU score is being calculated.")

    # Load model and tokenizer for Hugging Face model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Path for output file
    # output_file_path = os.path.join(output_folder, "bleu_results.json")

    # Read content from the ".de" file
    with open(de_file_path, 'r', encoding='utf-8') as de_file:
        de_content = de_file.read()

    # Read content from the ".en" file
    with open(en_file_path, 'r', encoding='utf-8') as en_file:
        en_content = en_file.read()

    # Split the content into sentences (assuming each line is a sentence)
    de_sentences = de_content.split('\n')
    en_sentences = en_content.split('\n')

    # Initialize total score
    total_bleu_score = 0
    bleu_score_data = {"scores": []}
    count = 0  # Number of successful BLEU score calculations

    # Loop through the sentences for translation and BLEU score calculation
    for i, (en_sentence, de_sentence) in enumerate(zip(en_sentences, de_sentences)):
        # Use Hugging Face to translate 'en' sentence to German
        response = get_simple_translation(model, tokenizer, en_sentence, target_language='de')

        reference = de_sentence  # 'de' is used as the human reference

        # Check if response or reference is not none
        if response and reference:
            # Calculation of the BLEU score for the response compared to the human reference
            bleu_score = sentence_bleu([reference.split()], response.split(),
                                       smoothing_function=SmoothingFunction().method1,
                                       weights=(1, 0))  # Calculate BLEU with unigram
            total_bleu_score += bleu_score  # Set total BLEU score

            # Formatting the BLEU score to two decimal places after the decimal point
            formatted_bleu_score = "{:.2f}".format(bleu_score)

            # Categorize BLEU score
            score_category = categorize_bleu_score(bleu_score)

            # Save the results as a dictionary
            bleu_score_data["scores"].append({
                "index": i + 1,
                "response": response,
                "reference": reference,
                "bleu_score": formatted_bleu_score,
                "score_category": score_category
            })
            count += 1

        # Stop loop if it has iterated `max_index` times
        if i >= max_index:
            break

    # Average BLEU score for the entire datasets
    average_bleu_score = round(total_bleu_score / count, 2)
    # Formatting the average BLEU score to two decimal places after the decimal point
    formatted_bleu_average_score = "{:.2f}".format(average_bleu_score)
    score_category = categorize_bleu_score(average_bleu_score)

    # # Save the results as JSON
    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     json.dump(bleu_score_data, output_file, indent=2, ensure_ascii=False)
    #
    # # Add BLEU average score to avg_results
    # with open(os.path.join(output_folder, "avg_results.json"), 'r', encoding='utf-8') as result_file:
    #     existing_data = json.load(result_file)

    # Add average BLEU score to data
    bleu_avg_data = {
        "BLEU": {
            "count": count,
            "average_bleu_score": formatted_bleu_average_score,
            "score_category": score_category
        }
    }
    # existing_data['Results'].append(bleu_avg_data)
    #
    # # Update avg_results.json file
    # with open(os.path.join(output_folder, "avg_results.json"), 'w', encoding='utf-8') as result_file:
    #     json.dump(existing_data, result_file, indent=4, ensure_ascii=False)

    print(average_bleu_score)
    # Create a metrics list for visualization
    metrics = [
        {"name": "Average BLEU Score", "score": float(average_bleu_score)},
        # Add more metrics if necessary
    ]
    return average_bleu_score, score_category


