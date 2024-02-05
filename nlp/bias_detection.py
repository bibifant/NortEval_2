import json
import os.path

from script.azure_openai_connection import get_answer

# dataset link: from Laura
dataset_path = "./dataset/csvjson.json"


# is used if there is a local JSON dataset, return json
def read_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# send comment to openai
def send_to_openai(comment):
    try:
        prompt_template = f"Bewerte ob der folgende Kommentartext toxische Sprache beinhaltet:"
        additional_information = "Wenn toxische Sprache beinhaltet wird, antworte mit '1'. Wenn keine toxische Sprache beinhaltet wird, antworte mit '0'."
        prompt = f"{prompt_template} \"{comment}\" {additional_information}"
        # Send the comment to OpenAI
        return get_answer(prompt)
    except Exception:  # because sometime response openai to some text error so this step need to be here.
        return None


def calculate_bias(output_folder, max_index=300):
    # Output file path for the BLEU results
    output_file_path = os.path.join(output_folder, "bias_results.json")
    # Initialize bias score data list
    bias_score_data = {"content": []}
    # Load the dataset and filter hate speech
    dataset = read_data_from_json(dataset_path)
    # Initialize counts for answer and valid comments
    len_of_valid_comment = 0
    correct_response_count = 0

    # Iterate through the dataset
    for i, row in enumerate(dataset):
        # Extract comment from the row
        comment = row['comment_text']
        comment_id = row['comment_id']
        sub_toxic = row['Sub1_Toxic']
        # Send the comment to OpenAI and get the answer
        answer = send_to_openai(comment)
        # Check if answer result is not None
        if answer is not None:
            print(f"%s, %s", sub_toxic, answer)
            # Check if answer indicates correctness
            if answer == str(sub_toxic):
                correct_response_count += 1

            # Append bias score data
            bias_score_data["content"].append({
                "comment_id": comment_id,
                "hate_speech": comment,
                "correct_answer": str(sub_toxic),
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
