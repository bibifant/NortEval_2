import spacy
from nltk.stem.snowball import SnowballStemmer

# Initialize a stemmer for the German language
stemmer = SnowballStemmer("german")

# Load a large-scale SpaCy model for German language for NLP tasks
nlp = spacy.load("de_core_news_lg")

def stem_words(words):
    # Stem a list of words using NLTK Snowball Stemmer.
    return [stemmer.stem(word) for word in words]

def extract_relevant_keywords(text, top_n=6):
    """
      Extract relevant keywords from text.
      :param text: String, text from which to extract keywords.
      :param top_n: Integer, number of top keywords to return.
      :return: List of stemmed top N keywords.
      """
    # Process the text through SpaCy NLP pipeline
    doc = nlp(text)

    # Extract lemmas of nouns, adjectives, and verbs that are not stop words
    relevant_words = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop]

    # Stem the relevant words
    stemmed_words = stem_words(relevant_words)

    # Return top N stemmed words
    return stemmed_words[:top_n]

def check_prompt_keywords_in_response(prompt_keywords, response_text):
    """
      Check the presence of prompt keywords in the response text.
      :param prompt_keywords: List of keywords extracted from the prompt.
      :param response_text: String, the response text to be checked.
      :return: Float, ratio of prompt keywords present in the response.
      """
    response_doc = nlp(response_text)
    response_words = [token.text for token in response_doc if not token.is_stop]
    response_stemmed = set(stem_words(response_words))

    # Calculate the proportion of prompt keywords present in the response
    prompt_keywords_present = sum(stem in response_stemmed for stem in prompt_keywords)
    return prompt_keywords_present / len(prompt_keywords)


# prompt = "Wie könnte die Welt im Jahr 2050 aussehen, wenn erneuerbare Energien die Hauptenergiequelle sind?"
# response = "Im Jahr 2050 könnte die Welt eine sehr unterschiedliche Energieversorgung aufweisen, wenn erneuerbare Energien die Hauptenergiequelle sind. Hier sind einige mögliche Szenarien:\n\n1. Elektrifizierung des Transportwesens: Da erneuerbare Energien wie Solarenergie und Windenergie von Natur aus intermittierend sind und nicht immer verfügbar sind, könnte die Elektrifizierung des Transportwesens eine Möglichkeit sein, um überschüssige Energie zu nutzen. Elektrofahrzeuge könnten mit erneuerbarer Energie geladen werden und so fossile Brennstoffe ersetzen.\n\n2. Dezentrale Energieversorgung: Eine weit verbreitete Nutzung von erneuerbaren Energien könnte dazu führen, dass die Energieversorgung dezentralisiert wird. Haushalte, Unternehmen und Gemeinden könnten ihre eigene Energie erzeugen und speichern, um unabhängiger von großen Energieversorgern zu sein.\n\n3. Wasserstoff als Energiespeicher: Wasserstoff könnte als Energiespeicher verwendet werden, um überschüssige Energie aus erneuerbaren Quellen zu speichern und bei Bedarf zu nutzen. Wasserstoff könnte auch als Brennstoff für Fahrzeuge und zur Wärmeerzeugung verwendet werden.\n\n4. Neue Arbeitsplätze und Wirtschaftssektoren: Eine Umstellung auf erneuerbare Energien könnte neue Arbeitsplätze und Wirtschaftssektoren schaffen. Es könnten neue Technologien und Infrastrukturen entstehen, um erneuerbare Energien zu nutzen, zu speichern und zu verteilen.\n\n5. Weniger Umweltverschmutzung: Eine Welt, die hauptsächlich auf erneuerbare Energien setzt, würde wahrscheinlich weniger Umweltverschmutzung und Treibhausgasemissionen aufweisen als eine, die sich auf fossile Brennstoffe verlässt. Dies könnte dazu beitragen, den Klimawandel zu bekämpfen und die Luft- und Wasserqualität zu verbessern.\n\nInsgesamt könnte eine Zukunft, die auf erneuerbaren Energien basiert, eine sauberere, dezentralisierte und nachhaltigere Welt schaffen."

# Schlüsselwortanalyse und Score-Berechnung
# prompt_keywords = extract_relevant_keywords(prompt)
# prompt_keywords_in_response_score = check_prompt_keywords_in_response(prompt_keywords, response)

# print("Prompt Schlüsselwörter:", prompt_keywords)
# print("Präsenz der Prompt-Schlüsselwörter in der Antwort:", prompt_keywords_in_response_score)

