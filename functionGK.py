import spacy

nlp = spacy.load("de_core_news_sm")

def upper_lower_case(text):
    # Verarbeiten des Textes mit spaCy
    doc = nlp(text)

    # Konvertiere alle Wörter im Text zu Kleinbuchstaben
    text_klein = text.lower()

    # Extrahieren aller Nomen
    nomen_klein = [token.text for token in nlp(text_klein) if token.pos_ in ('NOUN', 'PROPN')]

    # Sammeln aller kleingeschriebenen Wörter
    kleingeschrieben_worte = [token.text for token in doc if token.text.isalpha() and token.text.lower() not in nomen_klein and not token.is_sent_start]

    # Prozent berechnen
    prozentsatz_nomen = (sum(1 for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.text.lower() in nomen_klein and token.text.istitle()) / len(nomen_klein)) * 100
    prozentsatz_klein = (sum(1 for token in doc if token.text.isalpha() and token.text.lower() not in nomen_klein and not token.is_sent_start and token.text.islower()) / len(kleingeschrieben_worte)) * 100
    prozentsatz_satzanfang = (sum(1 for token in doc if token.is_sent_start and token.text.istitle()) / len(list(doc.sents))) * 100

    gesamt_prozentsatz = (prozentsatz_nomen + prozentsatz_klein + prozentsatz_satzanfang) / 3

    print("Es sind ", gesamt_prozentsatz, "% des Textes in der korrekten Groß-und Kleinschreibung.")

# Beispieltext
text_beispiel = "Der schnelle braune Fuchs springt über den faulen Hund und der Fuchs freut sich. Der große Baum."
upper_lower_case(text_beispiel)