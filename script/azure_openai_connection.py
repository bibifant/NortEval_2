import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_answer(prompt: str, max_response_tokens:int = 200, user_text: str = None):

    if user_text is not None:
        prompt.format(user_text)

    system_message = [{"role": "system", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt35", # model = "deployment_name".
        messages=system_message,
        temperature=0.7,
        max_tokens=max_response_tokens
    )

    return response.choices[0].message.content