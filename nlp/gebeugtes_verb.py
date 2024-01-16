# Check whether a conjugated verb is contained in the sentence. As the verb is the anchor of a German sentence,
# one must be included.

import spacy
from script.azure_openai_connection import get_answer

# global variable
nlp = spacy.load("de_core_news_sm")


def get_number(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            number_list = token.morph.get('Number')
            if number_list:
                number = number_list[0]
                return number


def get_person(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            person_list = token.morph.get("Person")
            person = person_list[0]
            return person


def get_nomen_numerus(sentence):
    doc = nlp(sentence)

    # Check if the word is a plural noun
    for token in doc:
        if token.pos_ == "NOUN":
            numerus_list = token.morph.get("Number")
            numerus = numerus_list[0]
            return numerus


def run_gebeugtes_verb_test(sentence):
    doc = nlp(sentence)
    gefundene_verben = []

    # Überprüfen, ob ein gebeugtes Verb im Satz vorhanden ist
    for token in doc:
        if token.pos_ == "VERB":
            number = get_number(sentence)
            numerus = get_nomen_numerus(sentence)

            if number == numerus and number:
                print(f"{sentence}: Dieser Satz enthält das gebeugte Verb {token.text}")
                gefundene_verben.append(token.text)
            elif number == "Sing":
                print(f"{sentence}: Dieser Satz enthält das gebeugte Verb {token.text}")
                gefundene_verben.append(token.text)

    if not gefundene_verben:
        print(f"{sentence}: Dieser Satz enthält kein gebeugtes Verb.")
    else:
        print(f"Gefundene Verben: {', '.join(gefundene_verben)}")
        return True

    return False


prompt = "Schreibe einen deutschen Satz mit mehreren gebeugten Verben:"
response = get_answer(prompt)

# Überprüfung ob der Satz ein gebeugtes Verb enthält
run_gebeugtes_verb_test(response)
