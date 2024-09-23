import json
import Levenshtein
from transformers import AutoModelForSequenceClassification, T5ForConditionalGeneration

from model_connection import get_answer

prompt_template = "Categorize which sentiment the example word contains: "
prompt_additional_instructions = "Respond in all lower caps and in a single word."
ds_json_file_path = "dataset/word_connotation_ds.json"


def load_data(ds_json_file_path):
    with open(ds_json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get("Wörter", [])


def generate_prompt(prompt_template, word):
    return f"{prompt_template} '{word}', 'very positive', 'positive', 'neutral', 'negative' or 'very negative'.{prompt_additional_instructions}"


def create_json_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def check_sentiment_match_exact(response_sentiment, reference_sentiment, allowed_distance=1):
    # If response_sentiment is a list, consider the first element
    if isinstance(response_sentiment, list):
        response_sentiment_lower = response_sentiment[0].lower()
    else:
        response_sentiment_lower = str(response_sentiment).lower()

    # Convert reference_sentiment to lowercase for case-insensitive comparison
    reference_sentiment_lower = reference_sentiment.lower()

    # Check if the response sentiment exactly matches the reference sentiment
    exact_match = response_sentiment_lower == reference_sentiment_lower

    # Check if the Levenshtein distance is within the allowed threshold
    distance = Levenshtein.distance(response_sentiment_lower, reference_sentiment_lower)
    within_distance_threshold = distance <= allowed_distance

    # Consider it a match if either the exact match or within_distance_threshold is true
    return exact_match or within_distance_threshold


def check_sentiment_match_in_category(response_sentiment, reference_sentiment):
    # Define sentiment categories
    negative_categories = ['negative', 'very negative']
    positive_categories = ['neutral', 'positive', 'very positive']
    neutral_categories = ['neutral']

    # If response_sentiment is a list, consider the first element
    if isinstance(response_sentiment, list):
        response_sentiment_lower = response_sentiment[0].lower()
    else:
        response_sentiment_lower = str(response_sentiment).lower()

    # Convert reference_sentiment to lowercase for case-insensitive comparison
    reference_sentiment_lower = reference_sentiment.lower()

    # Check if the model recognized the general sentiment correctly
    if reference_sentiment_lower in negative_categories:
        return response_sentiment_lower in negative_categories
    elif reference_sentiment_lower in positive_categories:
        return response_sentiment_lower in positive_categories
    elif reference_sentiment_lower in neutral_categories:
        return response_sentiment_lower in neutral_categories
    else:
        return False


def run_word_connotation_recognition(model, tokenizer):
    # Check model compatibility
    if not isinstance(model, (AutoModelForSequenceClassification, T5ForConditionalGeneration)):
        return None  # or raise an exception
    print(f"Word Connotation Recognition is running.")
    words_data = load_data(ds_json_file_path)
    results = []
    correct_exact_matches = 0
    correct_category_matches = 0

    for word_info in words_data:
        word = word_info.get("Wort", "")
        reference_sentiment = word_info.get("Sentiment", "")

        # Generate prompt for the Hugging Face model
        prompt = generate_prompt(prompt_template, word)

        # Tokenize input prompt using the Hugging Face tokenizer
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        # Generate response using the Hugging Face model
        outputs = model.generate(**inputs, max_new_tokens=50)

        # Decode the model's output
        sentiment_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Check if the sentiment matches the reference sentiment exactly or in category
        exact_match = check_sentiment_match_exact(sentiment_response, reference_sentiment)
        category_match = check_sentiment_match_in_category(sentiment_response, reference_sentiment)

        # Collect results
        result_entry = {
            'word': word,
            'prompt': prompt,
            'response': sentiment_response,
            'reference_sentiment': reference_sentiment,
            'exact_match': exact_match,
            'category_match': category_match
        }
        results.append(result_entry)

        if exact_match:
            correct_exact_matches += 1
        if category_match:
            correct_category_matches += 1

    total_entries = len(words_data)
    percentage_exact_matches = round((correct_exact_matches / total_entries), 2)
    percentage_category_matches = round((correct_category_matches / total_entries), 2)

    def categorize_results(percentage_matches, threshold_good=85, threshold_bad=65):
        if percentage_matches >= threshold_good:
            return "good"
        elif percentage_matches >= threshold_bad:
            return "average"
        else:
            return "bad"

    result_category_exact_match = categorize_results(percentage_exact_matches, 0.7, 0.5)
    result_category_category_match = categorize_results(percentage_category_matches)

    return percentage_category_matches

