from create_results import create_results
from metrics.bleu import calculate_bleu
from metrics.rouge import run_rouge
from nlp.bias_detection.hate_speech_detection import run_hate_speech
from nlp.bias_detection.word_connotation_recognition import run_word_connotation_recognition
from nlp.natural_language_quality_tests.natural_language_quality_assessor import run_natural_language_quality_assessor
from nlp.de_en import run_language_percentage
from nlp.upper_lower_case import run_upper_lower_case
from nlp.contains_verb import run_contains_verb
from nlp.sentiment_analysis_user_comments import run_sentiment_analysis


def main():

    # Create the results folder with timestamp
    output_folder = create_results()

    # Bleu
    calculate_bleu(output_folder)

    # Rouge
    run_rouge(output_folder)

    # Hate Speech
    run_hate_speech(output_folder)

    # Word Connotation Recognition
    run_word_connotation_recognition(output_folder)

    # Sentiment Analysis
    run_sentiment_analysis(output_folder)

    # Natural Language/ Plausibility
    run_natural_language_quality_assessor(output_folder)

    # Percentage of German & English words
    run_language_percentage(output_folder)

    # Upper Lower Case
    run_upper_lower_case(output_folder)

    # Contains Verb
    run_contains_verb(output_folder)



if __name__ == "__main__":
    main()
