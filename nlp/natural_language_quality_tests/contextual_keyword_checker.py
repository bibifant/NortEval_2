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