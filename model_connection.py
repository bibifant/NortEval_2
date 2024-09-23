from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, AutoConfig

from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, AutoConfig

from transformers import AutoModelForSeq2SeqLM, AutoModel, AutoTokenizer, AutoConfig
from transformers.models.bert import BertModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertModel,
    T5ForConditionalGeneration
)


def load_model_and_tokenizer(model_name_or_object):
    """
    Load the model and tokenizer based on the input, which can be either a model name (string) or a pre-loaded model object.

    :param model_name_or_object: Either a model name string or an initialized model object.
    :return: A tuple of (model, tokenizer).
    """

    # Debugging: Print the type of model_name_or_object
    print(f"Received input of type: {type(model_name_or_object)}")

    if isinstance(model_name_or_object, str):
        # Load model and tokenizer using the model name
        config = AutoConfig.from_pretrained(model_name_or_object)

        # Determine the correct model class based on the configuration
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_object)
        else:
            model = AutoModel.from_pretrained(model_name_or_object)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_object)
        print(f"Model and tokenizer loaded successfully from model name: {model_name_or_object}")

    elif isinstance(model_name_or_object, (AutoModelForSeq2SeqLM, AutoModel, BertModel, T5ForConditionalGeneration)):
        # If a model object is passed, load the corresponding tokenizer
        model = model_name_or_object
        tokenizer_name = model.config._name_or_path  # Extract the model name/path from the config
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Model and tokenizer loaded successfully from model object: {tokenizer_name}")

    else:
        raise ValueError(f"Unsupported model type or input. Received: {type(model_name_or_object)}")

    return model, tokenizer


def get_answer(model, tokenizer, prompt: str, max_response_tokens: int = 50, user_text: str = None):
    if user_text is not None:
        prompt = prompt.format(user_text)

    # Tokenizing des Inputs
    input_ids = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).input_ids
    # Response generieren
    output = model.generate(input_ids, max_length=max_response_tokens)

    # Response zurückgeben
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def get_simple_translation(model, tokenizer, text: str, target_language: str = 'de', max_response_tokens: int = 50):
    # Zielsprache setzen
    task_prefix = f"translate English to {target_language}: "

    prompt = task_prefix + text
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # response generieren
    output_ids = model.generate(input_ids, max_length=max_response_tokens)

    # response zurückgeben
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text
