import spacy
from langdetect import detect
nlp = spacy.load("en_core_web_sm")
def language_percentage(text):

    # Verarbeiten des gesamten Textes mit dem Spacy-Modell
    doc = nlp(text)

    # Zählen der englischen und deutschen Wörter
    english_words = sum(1 for token in doc if token.is_alpha and detect(token.text) == 'en')
    german_words = sum(1 for token in doc if token.is_alpha and detect(token.text) == 'de')

    # Zählen der Gesamtanzahl der Wörter
    total_words = sum(1 for token in doc if token.is_alpha)

    # Vermeiden der Division durch Null
    if total_words == 0:
        return 0, 0

    # Berechnung der Prozentwerte
    english_percentage = (english_words / total_words) * 100
    german_percentage = (german_words / total_words) * 100

    # Anpassung Prozentwert auf insgesamt 100%
    if english_percentage + german_percentage != 100:
        scale_factor = 100 / (english_percentage + german_percentage)
        english_percentage *= scale_factor
        german_percentage *= scale_factor

    print(f"Text: {doc}")
    print(f"Englische Wortanteil: {english_percentage:.2f}%")
    print(f"Deutsche Wortanteil: {german_percentage:.2f}%")

# Beispieltext
example_text = "Dies ist an example text in English und gutes Deutsch."

language_percentage(example_text)