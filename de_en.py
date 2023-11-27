import spacy
from langdetect import detect
nlp = spacy.load("de_core_news_sm")
def language_percentage(text):
    # Sprache des gesamten Textes bestimmen
    main_language = detect(text)

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
    print(f"Hauptsprachanteil des Textes: {main_language}")
    print(f"Englische Wortanteil: {english_percentage:.2f}%")
    print(f"Deutsche Wortanteil: {german_percentage:.2f}% \n")

# Beispieltext
example_text = "Dies ist an english example text und ein deutscher Text zum Überprüfen."
language_percentage(example_text)

example_text ="this is an example text in English."
language_percentage(example_text)

example_text = "Dies ist ein deutscher Beispieltext."
language_percentage(example_text)