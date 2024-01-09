from azure_openai_connection import get_answer

sentence ="El grupo HTW-Nortal está generando un gran proyecto!"

prompt = """
Translate {0} to german
""".format(sentence)

print(get_answer(prompt))