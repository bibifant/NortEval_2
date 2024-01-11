import requests
import json
from dotenv import load_dotenv
import os
from translate import Translator

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
BLEU_API = os.getenv("BLEU_DATASET_API_URL")


# verbinden mit Dataset mit nur einfach api ul
def fetch_dataset_from_api(url=BLEU_API):
    res = requests.get(url)
    if res.status_code == 200:
        return json.loads(res.text)
    else:
        raise Exception(f"API-Anfrage fehlgeschlagen mit Statuscode {res.status_code}.")


# von aktuelle Dataset nach bestimmte Attributen nehmen
def extract_translations(dataset):
    translations = []
    for row in dataset['rows']:
        translation = row['row']['translation']
        de_translation = translation.get('de', '')
        en_translation = translation.get('en', '')
        if de_translation and en_translation:
            translations.append({'de': de_translation, 'en': en_translation})
    return translations


# eine alternative Translator
def translate_texts(text, target_language='de'):
    translator = Translator(to_lang=target_language)
    translations = translator.translate(text)
    return translations


# wird nur benutzt, wenn da eine locale Dataset gibt
def read_data_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data