# from huggingface_hub import ModelCard
# import json
# import Hugginface_Model_Scanner
#
# model_names = Hugginface_Model_Scanner.generate_list_of_available_german_models()
# # 'mistralai/Mixtral-8x7B-Instruct-v0.1'
# model_name = model_names[1]
# card = ModelCard.load(model_name)
# card_dict = card.data.to_dict()
#
#
# languages = card_dict.get("language")
# print("Languages:", languages)
#
# if languages and "de" in languages:
#     print("The dictionary contains the language 'de'.")
#     card_json = json.dumps(card_dict)
#     file_path = f'model_card_{model_name.replace("/", "_")}.json'
#     with open(file_path, 'w') as json_file:
#         json_file.write(card_json)
#
#     print(f"Model card JSON has been written to {file_path}")
# else:
#     print("The dictionary does not contain the language 'de'.")


from huggingface_hub import ModelCard, HfApi
import json
import Hugginface_Model_Scanner


def get_model_popularity(model_name):
    api = HfApi()
    model_info = api.model_info(model_name)
    downloads_api = model_info.downloads
    likes_api = model_info.likes
    return downloads_api, likes_api


def save_model_card_if_language_present(model_name, language_code='de'):
    card = ModelCard.load(model_name)
    card_dict = card.data.to_dict()

    languages = card_dict.get("language")
    print("Languages:", languages)

    if languages and language_code in languages:
        print(f"The dictionary contains the language '{language_code}'.")
        card_json = json.dumps(card_dict)
        file_path = f'model_card_{model_name.replace("/", "_")}.json'
        with open(file_path, 'w') as json_file:
            json_file.write(card_json)

        print(f"Model card JSON has been written to {file_path}")
    else:
        print(f"The dictionary does not contain the language '{language_code}'.")


# Beispielhafte Nutzung:
model_names = Hugginface_Model_Scanner.generate_list_of_available_german_models()

# Erstelle eine Liste mit Modellnamen und deren Popularität
models_with_popularity = []
for name in model_names:
    downloads, likes = get_model_popularity(name)
    models_with_popularity.append((name, downloads, likes))

# Sortiere die Modelle nach der Anzahl der Downloads (absteigend)
models_with_popularity.sort(key=lambda x: x[1], reverse=True)

# Speichere die Model Cards der Modelle, die die Sprache 'de' enthalten
for name, downloads, likes in models_with_popularity:
    save_model_card_if_language_present(name, 'de')
    print(f"Model: {name}, Downloads: {downloads}, Likes: {likes}")

