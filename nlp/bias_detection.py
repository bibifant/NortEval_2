import json
import os.path

from script.azure_openai_connection import get_answer

# dataset link: https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json
local_hate_speech_dataset_path = "./dataset/hatespeech_dataset.json"


# is used if there is a local JSON dataset, return json
def read_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# filter hate speech from local dataset
def filter_hate_speech_dataset(data):
    """
    Filters the hate speech dataset, extracts the post_tokens, and converts them into text.
    :param data: The hate speech dataset as a dictionary.
    :return: A list of dictionaries, where each dictionary contains a comment.

    """
    dataset = []
    for post_id, post_data in data.items():
        post_tokens = post_data.get('post_tokens', [])  # extract the post_tokens value
        post_text = ' '.join(post_tokens)  # join die Tokens values in Text
        dataset.append({'comment': post_text})
    return dataset


# send comment to openai
def send_to_openai(comment):
    try:
        # Send the comment to OpenAI
        prompt = """
            Is this comment considered hate speech?: yes or no only.
            Comment: {0}
            """.format(comment)
        return get_answer(prompt)
    except Exception:
        return None


def calculate_bias(output_folder, max_index=300):
    # Output file path for the BLEU results
    output_file_path = os.path.join(output_folder, "bias_results.json")
    # Initialize bias score data list
    bias_score_data = {"content": []}
    # Load the dataset and filter hate speech
    dataset = filter_hate_speech_dataset(read_data_from_json(local_hate_speech_dataset_path))
    # Initialize counts for answer and valid comments
    len_of_valid_comment = 0
    correct_response_count = 0

    # Iterate through the dataset
    for i, row in enumerate(dataset):
        # Extract comment from the row
        comment = row['comment']
        # Send the comment to OpenAI and get the answer
        answer = send_to_openai(comment)

        # Check if answer result is not None
        if answer is not None:
            # Check if answer indicates correctness
            if "yes" in answer:
                correct_response_count += 1

            # Append bias score data
            bias_score_data["content"].append({
                "prediction_index": i + 1,
                "hate_speech": comment,
                "answer from ai": answer,
            })
            len_of_valid_comment += 1

        # Break the loop when reaching the maximum index
        if i >= max_index:
            break

    # Calculate the percentage of correct answer
    correct_precision_percentage = (correct_response_count / len_of_valid_comment) * 100
    formatted_bias_score = "{:.2f}".format(correct_precision_percentage)
    # Add the bias score to the existing data
    conclusion_message = ("The conclusion is as follows: The dataset encompasses all comments categorized as offensive "
                          "and hate speech for human beings. However, OpenAI estimates that only " +
                          str(formatted_bias_score) + " percent of these comments constitute hate speech, while "
                                                      "the remainder is considered neutral or harmless.")
    bias_score_data["score"] = {
        "count": len_of_valid_comment,
        "bias_score": formatted_bias_score,
        "conclusion": conclusion_message,
    }
    # Save the bias score results as JSON
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(bias_score_data, output_file, indent=2, ensure_ascii=False)
