import json

import requests

from azure_openai_connection import get_answer

dataset_url = "https://datasets-server.huggingface.co/rows?dataset=germeval_14&config=germeval_14&split=train&offset=0&length=100"
local_hate_speech_dataset_path = "./dataset/hatespeech_dataset.json"


# connect to dataset with just api url
def fetch_dataset_from_api(url):
    res = requests.get(url)
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        raise Exception(f"API request failed with status code {res.status_code}.")


# is used if there is a local JSON dataset, return json
def read_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# filter german dataset
def filter_dataset(data):
    dataset = []
    for row in data['rows']:  # loop in data['rows']
        comment_tokens = row['row']['tokens']  # get comment_tokens value
        comment_text = ' '.join(comment_tokens)  # make tokens to text
        comment_id = row['row']['id']  # get tokens_id as comment_id
        dataset.append(
            {'id': comment_id, 'comment': comment_text})  # add to values to list with key: 'id' and 'comment'
    return dataset


# filter hate speech from local dataset
def filter_hate_speech_dataset(data):
    dataset = []
    for post_id, post_data in data.items():
        post_tokens = post_data.get('post_tokens', [])  # Extrahiere die post_tokens, falls vorhanden
        post_text = ' '.join(post_tokens)  # Wandle die Tokens in Text um
        dataset.append({'comment': post_text})
    return dataset


# send comment to openai using hate speech dataset.
def check_if_hate_speech_via_openai(comment):
    # Send the comment to OpenAI
    prompt = """
    Question: is this comment hate speech?
    comment: {0} 
    answer only: yes or no!
    """.format(comment)
    return get_answer(prompt)


# send comment to openai from hugging face dataset
def send_to_openai(comment):
    # Send the comment to OpenAI
    prompt = """
    Provide one of the following categories: toxic, engaging, factclaiming.
    Comment: {0}
    """.format(comment)
    result = get_answer(prompt)
    return result


# inspire from: https://github.com/dslaborg/germeval2021/blob/master/experiments/experiment_7/experiment_7.py#L75
# def calc_f1_score_germeval(ly_true, ly_pred):
#     macro_f1 = 0
#     if len(ly_true.shape) == 1:
#         ly_true = ly_true[:, np.newaxis]
#         ly_pred = ly_pred[:, np.newaxis]
#     for i in range(ly_true.shape[1]):
#         report = classification_report(ly_true[:, i], ly_pred[:, i], output_dict=True)
#         precision_score = report['macro avg']['precision']
#         recall_score = report['macro avg']['recall']
#         lf1_score = 0
#         if precision_score + recall_score > 0:
#             lf1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
#         macro_f1 += lf1_score
#     return macro_f1 / ly_true.shape[1]


def calculate_score():
    # Load the datasets
    dataset = filter_dataset(fetch_dataset_from_api(dataset_url))

    # data = filter_hate_speech_dataset(read_data_from_json(local_hate_speech_dataset_path)) # get data from local dataset
    # Revise the code to perform the desired steps
    count = 0
    # For each row in the test dataset
    for row in dataset:
        count += 1
        # Filter out the comment (assuming the comment is in the 'text' column)
        comment = row['comment']

        # Send the comment to OpenAI and get the results
        result = send_to_openai(comment)
        print(result)
        if count >= 50:
            break
        # Assign the results to the corresponding categories (toxic, engaging, or factclaiming)
    #     # Analyze the results and categorize accordingly

    # Add the results to the respective categories here

    # Calculate and save the scores

    # Save the results and scores as desired


if __name__ == '__main__':
    # Call the calculate_score function
    calculate_score()
