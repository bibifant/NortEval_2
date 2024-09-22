from huggingface_hub import HfApi, ModelCard
import json

api = HfApi()

batch_size = 50
max_models_to_check = 50

german_models = []


def is_language_model(model_card):
    # Check if the model is tagged as a language model
    if "tags" in model_card:
        return any(tag in model_card.tags for tag in ["text-generation", "text2text-generation", "sequence-classification", "summarization", "translation"])
    return False


def generate_list_of_available_german_models():
    total_models_processed = 0
    while total_models_processed < max_models_to_check:
        models = list(api.list_models(limit=batch_size, full=False))

        if not models:
            break

        # Iterate through the models and check if they are trained in German
        for model_info in models:
            model_name = model_info.id
            try:
                # Load the model card
                card = ModelCard.load(model_name)
                card_dict = card.data.to_dict()
                languages = card_dict.get("language")

                # Check if "de" (German) is in the list of languages
                if languages and "de" in languages:
                    german_models.append(model_name)
                    print(f"The model '{model_name}' supports German.")
            except Exception as e:
                print(f"Could not load model card for {model_name}: {e}")

        total_models_processed += len(models)

        if len(models) < batch_size:
            break

    # Print all models that support German
    print("\nModels that support German:")
    for model in german_models:
        print(model)
    german_models_json = json.dumps(german_models)
    file_path = 'german-models.json'
    with open(file_path, 'w') as json_file:
        json_file.write(german_models_json)
    print(f"Model card JSON has been written to {file_path}")
    return german_models


generate_list_of_available_german_models()
