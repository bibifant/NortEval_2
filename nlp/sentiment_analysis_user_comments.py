import json
import os
import Levenshtein
from model_connection import get_answer

prompt_template = "Consider the following social media comments. Categorize the main sentiment each comment contains: "
prompt_additional_instructions = "Respond in all lower caps and in a single word."
ds_json_file_path = "datasets/GermEval2017_Testset_diachronic.json"


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def create_json_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def generate_prompt(comment):
    global prompt_template
    return f"{prompt_template} '{comment}', 'positive', 'neutral', 'negative' or 'mixed'.{prompt_additional_instructions}"


def analyze_sentiment(comment):
    sentiment = "positive"  # Placeholder
    return sentiment


def update_results_file(output_folder, percentage_exact_matches, result_category):
    result_file_path = os.path.join(output_folder, "avg_results.json")

    # Load existing result file
    with open(result_file_path, 'r', encoding='utf-8') as result_file:
        existing_data = json.load(result_file)

    # Add average values
    existing_data["Results"].append({'Sentiment_Analysis_User_Comments': {
        'percentage_exact_sentiment_recognized': percentage_exact_matches,
        'result_category_exact_recognition': result_category,
    }})

    # Update results file
    with open(result_file_path, 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)


def run_sentiment_analysis(output_folder):
    print(f"Sentiment Analysis of User Comments is running.")
    data = load_data(ds_json_file_path)
    output_file_path = os.path.join(output_folder, "sentiment_analysis_user_comments_results.json")
    results = []
    correct_matches_counter = 0
    skipped_data_counter = 0
    for comment_info in data:
        comment = comment_info["text"]
        ground_truth_sentiment = comment_info["sentiment"]

        # Generate prompt for sentiment analysis
        prompt = generate_prompt(comment)

        try:
            predicted_sentiment = get_answer(prompt)

        except Exception as e:
            # print(f"comment: {comment} skipped due to OpenAI guidelines")
            skipped_data_counter = skipped_data_counter + 1
            # If an error occurs, skip this comment and continue to the next one
            continue
        # Check if predicted sentiment matches ground truth sentiment
        exact_match = check_sentiment_match(predicted_sentiment, ground_truth_sentiment)

        # Increment correct_matches if there is a match
        if exact_match:
            correct_matches_counter += 1

        # Analyze sentiment of the comment
        predicted_sentiment = analyze_sentiment(comment)

        # Check if predicted sentiment matches ground truth sentiment
        exact_match = check_sentiment_match(predicted_sentiment, ground_truth_sentiment)

        # Collect results
        result_entry = {
            "comment": comment,
            "prompt": prompt,
            "response_sentiment": predicted_sentiment,
            "ground_truth_sentiment": ground_truth_sentiment,
            "match": exact_match
        }
        results.append(result_entry)

        # Increment correct_matches if there is a match
        if exact_match:
            correct_matches_counter += 1

    # Calculate accuracy
    total_comments = len(data) - skipped_data_counter
    accuracy = round((correct_matches_counter / total_comments) * 100,2)

    # Categorize results
    if accuracy >= 50:
        result_category = "good"
    elif accuracy >= 35:
        result_category = "average"
    else:
        result_category = "bad"

    # Save results to a JSON file
    output_file_path = os.path.join(output_folder, "sentiment_analysis_results.json")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    # Update average results file
    update_results_file(output_folder, accuracy, result_category)

    # Return the results list
    return results


def check_sentiment_match(response_sentiment, reference_sentiment, allowed_distance=1):
    if isinstance(response_sentiment, list):
        response_sentiment_lower = response_sentiment[0].lower()
    else:
        response_sentiment_lower = str(response_sentiment).lower()

    reference_sentiment_lower = reference_sentiment.lower()

    exact_match = response_sentiment_lower == reference_sentiment_lower

    distance = Levenshtein.distance(response_sentiment_lower, reference_sentiment_lower)
    within_distance_threshold = distance <= allowed_distance

    return exact_match or within_distance_threshold


def save_results(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"results": data}, output_file, ensure_ascii=False, indent=2)

