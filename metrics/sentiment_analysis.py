import json
from script.azure_openai_connection import get_answer

prompt_template = "Categorize which sentiment the example word contains: "
prompt_additional_instructions = "Respond in all lower caps and in a single word."
json_file_path = "../dataset/sentiment_analysis_ds.json"
output_file_path = "sentiment_results.json"


def load_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data.get("Wörter", [])  # Assuming "Wörter" is the key for words in the JSON


def generate_prompt(prompt_template, word):
    return f"{prompt_template} '{word}', 'very positive', 'positive', 'neutral', 'negative' or 'very negative'.{prompt_additional_instructions}"


def create_json_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def run_sentiment_analysis(prompt_template, num_tokens, json_file_path, output_file_path):
    # Load data from JSON file
    words_data = load_data(json_file_path)

    results = []

    for word_info in words_data:
        word = word_info.get("Wort", "")  # Assuming "Wort" is the key for words in the JSON

        # Generate prompt and get sentiment analysis
        prompt = generate_prompt(prompt_template, word)
        sentiment = get_answer(prompt, num_tokens)  # Adjust as needed for your function

        print(f"The sentiment of the word '{word}' is: {sentiment}")

        # Collect results
        result_entry = {
            'word': word,
            'prompt': prompt,
            'response': sentiment
        }
        results.append(result_entry)

    # Create JSON file from the results
    create_json_file(results, output_file_path)


run_sentiment_analysis(prompt_template, 200, json_file_path, output_file_path)
