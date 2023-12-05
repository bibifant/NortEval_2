# Überprüfung ob ein gebeugtes Verb im Satz enthalten ist. Da das Verb der Anker eines deutschen Satzes ist, muss eins enthalten sein.

import spacy

nlp = spacy.load("de_core_news_sm")
satz1 = "Der Hund spielt im Park."
satz2 = "Die Vögel singen fröhlich."
satz3 = "lachen, spielen, singen."
satz4 = "Fenster singen im Park."


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


def enthält_gebeugtes_verb(satz):
    doc = nlp(satz)

    # Überprüfen, ob ein gebeugtes Verb im Satz vorhanden ist
    for token in doc:
        if (token.pos_ == "VERB"):
            if get_number(satz) == get_nomen_numerus(satz) and get_number(satz):
                print(f"{satz}: Dieser Satz enthält das gebeugte Verb {token}")
                return True
            if get_number(satz) == "Sing":
                print(f"{satz}: Dieser Satz enthält das gebeugte Verb {token}")
                return True
            else:
                print(f"{satz}: Dieser Satz enthält kein gebeugtes Verb.")
            return False


print(enthält_gebeugtes_verb(satz1))
print(enthält_gebeugtes_verb(satz2))
print(enthält_gebeugtes_verb(satz3))
print(enthält_gebeugtes_verb(satz4))