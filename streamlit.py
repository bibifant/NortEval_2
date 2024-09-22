# import streamlit as st
# import execute_selection
# import matplotlib.pyplot as plt
# from metrics.bleu import calculate_bleu
# from metrics.rouge import run_rouge
# from nlp.bias_detection import run_hate_speech,word_connotation_recognition
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
#
# from nlp.bias_detection.word_connotation_recognition import run_word_connotation_recognition
#
# # Static scores for comparison
# static_scores = {
#     "Average BLEU Score": 0.51,
#     "Average ROUGE-1 Score": 0.33,
#     "Average ROUGE-2 Score": 0.1,
#     "Average ROUGE-L Score": 0.16,
#     "Hate Speech Detection Score": 0.72,
#     "Word Connotation Recognition Score": 0.86,
# }
#
#
# def display_benchmark_results(metrics):
#     if isinstance(metrics, list):
#         st.write("Benchmark Results:")
#
#         # Extract model scores
#         metrics_names = [metric.get('name', 'Unnamed Metric') for metric in metrics]
#         metrics_scores = [metric.get('score', 0) for metric in metrics]
#
#         # Define ChatGPT static scores
#         chatgpt_metrics = {
#             "Average BLEU Score": 0.51,
#             "Average ROUGE-1 Score": 0.33,
#             "Average ROUGE-2 Score": 0.1,
#             "Average ROUGE-L Score": 0.16,
#             "Hate Speech Detection Score": 0.72,
#             "Word Connotation Recognition Score": 0.86,
#         }
#
#         # Prepare data for the bar chart
#         all_metrics_names = []
#         all_scores = []
#
#         for name in metrics_names:
#             all_metrics_names.append(name)
#             all_scores.append(metrics_scores[metrics_names.index(name)])
#
#             # Add corresponding ChatGPT scores
#             chatgpt_name = f"ChatGPT {name}"
#             all_metrics_names.append(chatgpt_name)
#             all_scores.append(chatgpt_metrics.get(name, 0))
#
#         # Create a bar chart
#         fig, ax = plt.subplots()
#         bar_width = 0.35  # Width of the bars
#         x = range(len(all_metrics_names))
#
#         # Create bars and ensure correct colors
#         bars = ax.bar(x, all_scores, width=bar_width,
#                       color=['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(all_scores))])
#
#         # Add labels and title
#         ax.set_ylabel('Score')
#         ax.set_title('Benchmark Results')
#         ax.set_xticks(x)
#         ax.set_xticklabels(all_metrics_names, rotation=45, ha='right')
#         ax.set_ylim(0, 1)  # Set y-axis limits (0 to 1)
#
#         # Adding a legend
#         ax.legend(['Model Score', 'ChatGPT Score'], loc='upper left')
#
#         st.pyplot(fig)
#     else:
#         st.error("Metrics data is not in the expected format.")
#
#
# # Load model and tokenizer from Hugging Face
# def load_model_and_tokenizer(model_name):
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return model, tokenizer
#
#
# # Main app logic
# def main():
#     st.title("Hugging Face Model Benchmark App")
#     st.write("Scan for available German language models from Hugging Face and benchmark them.")
#
#     # Initialize session state variables
#     if 'models_with_popularity' not in st.session_state:
#         st.session_state.models_with_popularity = None
#     if 'selected_model' not in st.session_state:
#         st.session_state.selected_model = None
#     if 'benchmark_done' not in st.session_state:
#         st.session_state.benchmark_done = False
#
#     # Button to trigger the model scan
#     if st.button('Start Model Scan'):
#         st.write("Scanning models...")
#         st.session_state.models_with_popularity = execute_selection.get_available_models()
#
#     # Check if models were found after scanning
#     if st.session_state.models_with_popularity is not None:
#         if st.session_state.models_with_popularity:
#             model_options = [
#                 f"{name} (Downloads: {downloads}, Likes: {likes})"
#                 for name, downloads, likes in st.session_state.models_with_popularity
#             ]
#
#             # Dropdown to select a model
#             st.session_state.selected_model = st.selectbox(
#                 "Please select a model:",
#                 model_options
#             )
#
#             # Extract model name from the selection
#             selected_model_name = st.session_state.selected_model.split(" ")[0]
#
#             # Display the benchmark button after model selection
#             if st.button('Run Benchmark'):
#                 st.write("Running benchmark...")
#
#                 # Load model and tokenizer
#                 if 'model' not in st.session_state:
#                     model, tokenizer = load_model_and_tokenizer(selected_model_name)
#                     st.session_state.model = model
#                     st.session_state.tokenizer = tokenizer
#                 else:
#                     model = st.session_state.model
#                     tokenizer = st.session_state.tokenizer
#
#                 # Calculate BLEU score
#                 st.write("Calculating BLEU score...")
#                 average_bleu_score, bleu_category = calculate_bleu(model_name=selected_model_name, max_index=10)
#                 st.write("BLEU score:", average_bleu_score)
#
#                 # Calculate ROUGE score
#                 st.write("Calculating ROUGE score...")
#                 avg_rouge1, avg_rouge2, avg_rougeL = run_rouge(model_name=selected_model_name, tokenizer=tokenizer, max_index=10)
#                 st.write("ROUGE score 1:", avg_rouge1)
#                 st.write("ROUGE score 2:", avg_rouge2)
#                 st.write("ROUGE score L:", avg_rougeL)
#
#
#
#                 # Run hate speech detection
#                 st.write("Running Hate Speech detection...")
#                 hate_speech_precision, hate_speech_category = run_hate_speech(model_name=selected_model_name, tokenizer=tokenizer, max_index=300)
#                 st.write("Hate Speech Recognition Precision:", hate_speech_precision)
#
#                 # Run word connotation recognition
#                 st.write("Running Word Connotation Recognition...")
#                 word_connotation_score = run_word_connotation_recognition(model, tokenizer)
#
#
#                 # Create a metrics list for visualization
#                 metrics = [
#                     {"name": "Average BLEU Score", "score": float(average_bleu_score)},
#                     {"name": "Average ROUGE-1 Score", "score": float(avg_rouge1)},
#                     {"name": "Average ROUGE-2 Score", "score": float(avg_rouge2)},
#                     {"name": "Average ROUGE-L Score", "score": float(avg_rougeL)},
#                     {"name": "Hate Speech Detection Score", "score": float(hate_speech_precision)},
#                     # Add this line
#                 ]
#                 # Check if the score is valid before adding to metrics
#                 if word_connotation_score is not None:
#                     metrics.append(
#                         {"name": "Word Connotation Recognition Score", "score": float(word_connotation_score)})
#                 else:
#                     st.write("Word Connotation Recognition could not be performed.")
#
#                 # Add static scores to metrics
#                 for name, score in static_scores.items():
#                     metrics.append({"name": f"ChatGPT {name}", "score": score})
#
#                 # Display benchmark results
#                 display_benchmark_results(metrics)
#
#                 st.session_state.benchmark_done = True
#         else:
#             st.error("Error 01: No models found.")
#     else:
#         st.error("Please start the scan to find models.")
#
# if __name__ == "__main__":
#     main()
import streamlit as st
import execute_selection
import matplotlib.pyplot as plt
from metrics.bleu import calculate_bleu
from metrics.rouge import run_rouge
from nlp.bias_detection import run_hate_speech, word_connotation_recognition
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nlp.bias_detection.word_connotation_recognition import run_word_connotation_recognition

# Static scores for comparison
static_scores = {
    "Average BLEU Score": 0.51,
    "Average ROUGE-1 Score": 0.33,
    "Average ROUGE-2 Score": 0.1,
    "Average ROUGE-L Score": 0.16,
    "Hate Speech Detection Score": 0.72,
    "Word Connotation Recognition Score": 0.86,
}


def display_benchmark_results(metrics):
    if isinstance(metrics, list):
        st.write("Benchmark Results:")

        # Extract model scores
        metrics_names = [metric.get('name', 'Unnamed Metric') for metric in metrics]
        metrics_scores = [metric.get('score', 0) for metric in metrics]

        # Prepare data for the bar chart
        all_metrics_names = []
        all_scores = []

        for name, score in zip(metrics_names, metrics_scores):
            all_metrics_names.append(name)
            all_scores.append(score)

            # Add corresponding ChatGPT scores
            chatgpt_name = f"ChatGPT {name}"
            all_metrics_names.append(chatgpt_name)
            all_scores.append(static_scores.get(name, 0))

        # Create a bar chart
        fig, ax = plt.subplots()
        bar_width = 0.35  # Width of the bars
        x = range(len(all_metrics_names))

        # Create bars
        bars = ax.bar(x, all_scores, width=bar_width, color=['skyblue' if i % 2 == 0 else 'lightgreen' for i in range(len(all_scores))])

        # Add labels and title
        ax.set_ylabel('Score')
        ax.set_title('Benchmark Results')
        ax.set_xticks(x)
        ax.set_xticklabels(all_metrics_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)  # Set y-axis limits (0 to 1)

        # Adding a legend
        ax.legend(['Model Score', 'ChatGPT Score'], loc='upper left')

        st.pyplot(fig)
    else:
        st.error("Metrics data is not in the expected format.")


# Load model and tokenizer from Hugging Face
def load_model_and_tokenizer(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# Main app logic
def main():
    st.title("Hugging Face Model Benchmark App")
    st.write("Scan for available German language models from Hugging Face and benchmark them.")

    # Initialize session state variables
    if 'models_with_popularity' not in st.session_state:
        st.session_state.models_with_popularity = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'benchmark_done' not in st.session_state:
        st.session_state.benchmark_done = False

    # Button to trigger the model scan
    if st.button('Start Model Scan'):
        st.write("Scanning models...")
        st.session_state.models_with_popularity = execute_selection.get_available_models()

    # Check if models were found after scanning
    if st.session_state.models_with_popularity is not None:
        if st.session_state.models_with_popularity:
            model_options = [
                f"{name} (Downloads: {downloads}, Likes: {likes})"
                for name, downloads, likes in st.session_state.models_with_popularity
            ]

            # Dropdown to select a model
            st.session_state.selected_model = st.selectbox(
                "Please select a model:",
                model_options
            )

            # Extract model name from the selection
            selected_model_name = st.session_state.selected_model.split(" ")[0]

            # Display the benchmark button after model selection
            if st.button('Run Benchmark'):
                st.write("Running benchmark...")

                # Load model and tokenizer
                model, tokenizer = load_model_and_tokenizer(selected_model_name)

                # Calculate BLEU score
                st.write("Calculating BLEU score...")
                average_bleu_score, bleu_category = calculate_bleu(model_name=selected_model_name, max_index=10)
                st.write("BLEU score:", average_bleu_score)

                # Calculate ROUGE score
                st.write("Calculating ROUGE score...")
                avg_rouge1, avg_rouge2, avg_rougeL = run_rouge(model_name=selected_model_name, tokenizer=tokenizer, max_index=10)
                st.write("ROUGE score 1:", avg_rouge1)
                st.write("ROUGE score 2:", avg_rouge2)
                st.write("ROUGE score L:", avg_rougeL)

                # Run hate speech detection
                st.write("Running Hate Speech detection...")
                hate_speech_precision, hate_speech_category = run_hate_speech(model_name=selected_model_name, tokenizer=tokenizer, max_index=300)
                st.write("Hate Speech Recognition Precision:", hate_speech_precision)

                # Run word connotation recognition
                st.write("Running Word Connotation Recognition...")
                word_connotation_score = run_word_connotation_recognition(model, tokenizer)

                # Create a metrics list for visualization
                metrics = [
                    {"name": "Average BLEU Score", "score": float(average_bleu_score)},
                    {"name": "Average ROUGE-1 Score", "score": float(avg_rouge1)},
                    {"name": "Average ROUGE-2 Score", "score": float(avg_rouge2)},
                    {"name": "Average ROUGE-L Score", "score": float(avg_rougeL)},
                    {"name": "Hate Speech Detection Score", "score": float(hate_speech_precision)},
                    {"name": "Word Connotation Recognition Score", "score": float(word_connotation_score) if word_connotation_score is not None else 0},
                ]

                # Display benchmark results
                display_benchmark_results(metrics)

                st.session_state.benchmark_done = True
        else:
            st.error("Error 01: No models found.")
    else:
        st.error("Please start the scan to find models.")

if __name__ == "__main__":
    main()
