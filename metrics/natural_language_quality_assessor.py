import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from math import exp
import json
import os
from script.azure_openai_connection  import get_answer
from metrics.bert_semantic_similarity import calculate_semantic_similarity
from metrics.contextual_keyword_checker import extract_relevant_keywords, check_prompt_keywords_in_response

# Path to JSON-file in 'dataset' folder
# You can change the prompts in the dataset according to your needs.
path_to_dataset = '../dataset/natural_language_dataset.json'


# Function to calculate the naturalness score of generated text
def calculate_naturalness_score(model, tokenizer, text):
    tokenize_input = tokenizer.tokenize(text)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model(tensor_input, labels=tensor_input).loss
    return exp(loss.item())


# Function to evaluate the quality of generated text responses
def evaluate_generated_text_quality(output_folder):
    # Load the model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    print(f"Natural language quality is being assessed.")

    # Define the path for the output file
    output_file_path = os.path.join(output_folder, "natural_language_results.json")

    # List of prompts for different text generation tasks
    with open('dataset/natural_language_dataset.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        prompts = data['prompts']

    current_results = []
    total_naturalness = 0
    total_semantic_similarity = 0
    total_keywords_in_response = 0

    for prompt in prompts:
        try:
            # Generate a response
            response_text = get_answer(prompt)

            # Calculate the naturalness score of the response
            naturalness_score = calculate_naturalness_score(model, tokenizer, response_text)

            # Calculate semantic similarity between prompt and response
            semantic_similarity = calculate_semantic_similarity(prompt, response_text)

            # Keyword analysis and calculation of keywords in response
            prompt_keywords = extract_relevant_keywords(prompt)
            keywords_in_response_score = check_prompt_keywords_in_response(prompt_keywords, response_text)

            # Accumulate scores for averaging
            total_naturalness += naturalness_score
            total_semantic_similarity += semantic_similarity
            total_keywords_in_response += keywords_in_response_score

            # Store the results for each prompt
            current_results.append({
                "Prompt": prompt,
                "Response": response_text,
                "Naturalness score": naturalness_score,
                "Semantic Similarity Score": semantic_similarity,
                "Prompt Keywords": prompt_keywords,
                "Keywords in Response Score": keywords_in_response_score
            })
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")

    # Calculate the average scores
    average_naturalness = total_naturalness / len(prompts) if prompts else 0
    average_similarity = total_semantic_similarity / len(prompts) if prompts else 0
    average_keywords_in_response = total_keywords_in_response / len(prompts) if prompts else 0

    # Write the results to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(current_results, output_file, ensure_ascii=False, indent=2)

    # Prepare data for storing the average scores
    avg_json_data = {"average_naturalness_score": average_naturalness, "average_semantic_similarity": average_similarity, "average_keywords_in_response_score": average_keywords_in_response}

    # Load existing results file
    with open(os.path.join(output_folder, "avg_results.json"), 'r', encoding='utf-8') as result_file:
            existing_data = json.load(result_file)

    # Add the average scores to the results file
    existing_data["Results"].append(avg_json_data)

    # Update the results file
    with open(os.path.join(output_folder, "avg_results.json"), 'w', encoding='utf-8') as result_file:
        json.dump(existing_data, result_file, ensure_ascii=False, indent=4)

    return current_results, avg_json_data

