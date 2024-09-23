# import json
# import os.path
#
# from openai_connection import get_answer
#
# # dataset link: from Laura
# dataset_path = "./datasets/hate_speech_germeval21_ds.json"
#
#
# # is used if there is a local JSON dataset, return json
# def read_data_from_json(file_path):
#     with open(file_path, "r") as f:
#         data = json.load(f)
#     return data
#
#
# def save_results(output_file_path, data):
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         json.dump({"hate_speech_results": data}, output_file, ensure_ascii=False, indent=2)
#
#
# def update_results_file(output_folder, precision_percentage, len_of_valid_comment, correct_response_count):
#     # Define file path
#     avg_result_file_path = os.path.join(output_folder, "avg_results.json")
#     # Prepare bias score data
#     bias_score_data = {"Hate Speech detection": {
#             "valid_comment_count": len_of_valid_comment,
#             "correct_answer_from_openai_count": correct_response_count,
#             "hate_speech_score": round(precision_percentage, 2),
#             "rating": categorize_hate_speech_detection(precision_percentage)
#     }}
#
#     # Load existing result file
#     with open(avg_result_file_path, 'r', encoding='utf-8') as result_file:
#         existing_data = json.load(result_file)
#     # Append bias score data to existing data
#     existing_data["Results"].append(bias_score_data)
#     # Update results file
#     with open(avg_result_file_path, 'w', encoding='utf-8') as result_file:
#         json.dump(existing_data, result_file, ensure_ascii=False, indent=2)
#
#
# def categorize_hate_speech_detection(precision_percentage, high_point=80, low_point=40):
#     if precision_percentage > high_point:
#         return "good"
#     elif low_point <= precision_percentage <= high_point:
#         return "average"
#     else:
#         return "bad"
#
#
# # send comment to openai
# def send_to_openai(comment):
#     try:
#         prompt_template = f"Bewerte ob der folgende Kommentartext toxische Sprache beinhaltet:"
#         additional_information = "Wenn toxische Sprache beinhaltet wird, antworte mit '1'. Wenn keine toxische Sprache beinhaltet wird, antworte mit '0'."
#         prompt = f"{prompt_template} \"{comment}\" {additional_information}"
#         # Send the comment to OpenAI
#         return get_answer(prompt)
#     except Exception:  # because sometime response openai to some text error so this step need to be here.
#         return None
#
#
# def run_hate_speech(output_folder, max_index=300):
#     print(f"Hate speech detection is running.")
#     # Output file path for the Bias results
#     output_file_path = os.path.join(output_folder, "hate_speech_results.json")
#     # Initialize bias score data list
#     bias_data = {"content": []}
#     # Load the dataset
#     dataset = read_data_from_json(dataset_path)
#     # Initialize counts for answer and valid comments
#     len_of_valid_comment = 0
#     correct_response_count = 0
#
#     # Iterate through the dataset
#     for i, row in enumerate(dataset):
#         # Extract comment from the row
#         comment = row['comment_text']
#         comment_id = row['comment_id']
#         sub_toxic = row['Sub1_Toxic']
#         # Send the comment to OpenAI and get the answer
#         answer = send_to_openai(comment)
#         # Check if answer result is not None
#         if answer is not None:
#             # Check if answer indicates correctness
#             if answer == str(sub_toxic):
#                 correct_response_count += 1
#
#             # Append bias score data
#             bias_data["content"].append({
#                 "comment_id": comment_id,
#                 "hate_speech": comment,
#                 "correct_answer": str(sub_toxic),
#                 "answer_from_ai": answer,
#             })
#             len_of_valid_comment += 1
#
#         # Break the loop when reaching the maximum index
#         if i >= max_index:
#             break
#
#     # Calculate the percentage of correct answer
#     precision_percentage = (correct_response_count / len_of_valid_comment) * 100
#
#     # Save the bias score results as JSON
#     save_results(output_file_path, bias_data)
#     # Update score data to avg_file
#     update_results_file(output_folder, precision_percentage, len_of_valid_comment, correct_response_count)
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Dataset path
dataset_path = "./datasets/hate_speech_germeval21_ds.json"


def read_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def categorize_hate_speech_detection(precision_percentage, high_point=80, low_point=40):
    if precision_percentage > high_point:
        return "good"
    elif low_point <= precision_percentage <= high_point:
        return "average"
    else:
        return "bad"


# Function to load model and tokenizer from Hugging Face
def load_model_and_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# New function to predict with a Hugging Face model
def predict_with_model(model, tokenizer, comment):
    inputs = tokenizer(comment,return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax(dim=-1).item()  # Assuming binary classification, 0 or 1
    return prediction


# Main hate speech detection function
def run_hate_speech(model_name, tokenizer,  max_index=300):
    print(f"Hate speech detection is running.")

    # Load model and tokenizer from Hugging Face
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Initialize counts for answer and valid comments
    len_of_valid_comment = 0
    correct_response_count = 0

    # Load the dataset
    dataset = read_data_from_json(dataset_path)

    # Iterate through the dataset
    for i, row in enumerate(dataset):
        # Extract comment and the labeled hate speech value (0 or 1)
        comment = row['comment_text']
        sub_toxic = row['Sub1_Toxic']

        # Predict using the Hugging Face model
        model_prediction = predict_with_model(model, tokenizer, comment)

        # Check if model's prediction is correct
        if model_prediction == sub_toxic:
            correct_response_count += 1

        len_of_valid_comment += 1

        # Break the loop when reaching the maximum index
        if i >= max_index:
            break

    # Calculate the percentage of correct answers
    if len_of_valid_comment > 0:
        precision_percentage = round(correct_response_count / len_of_valid_comment, 2)
    else:
        precision_percentage = 0

    # Return precision percentage and rating (good, average, bad)
    return precision_percentage, categorize_hate_speech_detection(precision_percentage)
