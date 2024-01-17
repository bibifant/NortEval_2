import requests
import json


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
