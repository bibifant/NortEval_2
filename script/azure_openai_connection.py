import os

import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def get_answer(prompt: str, max_response_tokens: int = 600, user_text: str = None):
    if user_text is not None:
        prompt.format(user_text)

    system_message = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt35",  # model = "deployment_name".
        messages=system_message,
        temperature=0.7,
        max_tokens=max_response_tokens
    )
    return response.choices[0].message.content


def get_simple_translation(text, target_language='de', max_response_tokens=200):
    try:
        neutral_prompt = "Translate the following text to {0}: {1}".format(target_language, text)
        response = client.chat.completions.create(
            model="gpt35",
            messages=[{"role": "user", "content": neutral_prompt}],
            temperature=0.7,
            max_tokens=max_response_tokens
        )

        # Extrahiere die übersetzte Nachricht aus der Antwort
        translated_text = response.choices[0].message.content
        return translated_text
    except Exception as e:
        print(f"Error: {e}")
        return None


"""
wenn der Text sensitive Information erhalten, wird bad request zurück gegeben
z.B: original_text: these mothers pump their first offspring full of this pollutant, and most of them die.
openai.BadRequestError: Error code: 400 - 
{'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. 
Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 
'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400}}
"""
