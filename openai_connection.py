import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("BENCHMARK-KEY")


def get_answer(prompt: str, max_response_tokens: int = 50, user_text: str = None):
    if user_text is not None:
        prompt.format(user_text)

    system_message = [{"role": "system", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # model = "deployment_name".
        messages=system_message,
        temperature=0.7,
        max_tokens=max_response_tokens
    )
    return response.choices[0].message.content


def get_simple_translation(text, target_language='de', max_response_tokens=50):
    try:
        neutral_prompt = "Translate the following text to {0}: {1}".format(target_language, text)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Update to OpenAI's model name
            messages=[{"role": "user", "content": neutral_prompt}],
            temperature=0.7,
            max_tokens=max_response_tokens
        )
        # Extrahiere die übersetzte Nachricht aus der Antwort
        translated_text = response.choices[0].message.content
        return translated_text
    except Exception as e:
        return None


"""
wenn der Text sensitive Information erhalten, wird bad request zurück gegeben
z.B: original_text: these mothers pump their first offspring full of this pollutant, and most of them die.
openai.BadRequestError: Error code: 400 - 
{'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. 
Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 
'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400}}
"""

import os
import openai
from dotenv import load_dotenv

# Load environment variables (make sure .env contains your OpenAI API Key)
load_dotenv()

# Set up the OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


def test_openai_connection():
    try:
        # Send a simple test prompt to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say hello in German."}],
            temperature=0.7,
            max_tokens=50  # Small response limit for testing
        )
        # Print the response from the model
        print("Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Execute the test function
if __name__ == "__main__":
    test_openai_connection()
