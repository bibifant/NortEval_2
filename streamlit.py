# import streamlit as st
# import execute_selection
# import matplotlib.pyplot as plt
# import json
# import os
# from metrics.bleu import calculate_bleu
# from metrics.rouge import run_rouge  # Ensure this import is correct
# from openai_connection import load_model_and_tokenizer  # Import the function to load model and tokenizer
#
# # Set the absolute path for results
# absolute_path = "/Users/lauri/htw/nortal-llm/"
#
# # Function to display benchmark results
# def display_benchmark_results(metrics):
#     st.write("Raw Metrics Data:", metrics)
#
#     if isinstance(metrics, list):
#         metrics_names = [metric.get('name', 'Unnamed Metric') for metric in metrics]
#         metrics_scores = [metric.get('score', 0) for metric in metrics]
#
#         # Create a bar chart
#         fig, ax = plt.subplots()
#         ax.bar(metrics_names, metrics_scores, color=['skyblue', 'lightgreen'])
#         ax.set_ylabel('Score')
#         ax.set_title('Benchmark Results')
#         # Set y-axis limits (0 to 1)
#         ax.set_ylim(0, 1)
#         plt.xticks(rotation=45, ha='right')
#         st.pyplot(fig)
#     else:
#         st.error("Metrics data is not in the expected format.")
#
# # Main app logic
# def main():
#     st.title("Hugging Face Modell Scanner")
#     st.write("Um verfügbare deutschsprachige LLMs auf Hugging Face zu finden, bitte den Scan starten.")
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
#     if st.button('Modell Scan jetzt starten'):
#         st.write("Modell Scan läuft...")
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
#                 "Bitte wählen Sie ein Modell aus:",
#                 model_options
#             )
#
#             # Extract model name from the selection
#             selected_model_name = st.session_state.selected_model.split(" ")[0]
#
#             # Display the benchmark button after model selection
#             if st.button('Benchmark ausführen'):
#                 st.write("Benchmark wird ausgeführt...")
#
#                 # Load model and tokenizer
#                 model, tokenizer = load_model_and_tokenizer(selected_model_name)
#
#                 # Calculate BLEU score
#                 average_bleu_score, score_category = calculate_bleu(model_name=selected_model_name, max_index=10)
#
#                 # Calculate ROUGE score
#                 avg_rouge1, avg_rouge2, avg_rougeL = run_rouge(model_name=selected_model_name, tokenizer=tokenizer, max_index=10)
#
#                 # Create a metrics list for visualization
#                 metrics = [
#                     {"name": "Average BLEU Score", "score": float(average_bleu_score)},
#                     {"name": "Average ROUGE-1 Score", "score": float(avg_rouge1)},
#                     {"name": "Average ROUGE-2 Score", "score": float(avg_rouge2)},
#                     {"name": "Average ROUGE-L Score", "score": float(avg_rougeL)}
#                 ]
#
#                 # Display benchmark results
#                 display_benchmark_results(metrics)
#
#                 st.session_state.benchmark_done = True
#         else:
#             st.error("Fehler 01: Keine Modelle gefunden.")
#     else:
#         st.error("Bitte den Scan starten, um Modelle zu finden.")
#
# if __name__ == "__main__":
#     main()


import streamlit as st
import execute_selection
import matplotlib.pyplot as plt
import json
import os
from metrics.bleu import calculate_bleu
from metrics.rouge import run_rouge
from openai_connection import load_model_and_tokenizer

# Set the absolute path for results
absolute_path = "/Users/lauri/htw/nortal-llm/"

# Function to display benchmark results
def display_benchmark_results(metrics):
    st.write("Raw Metrics Data:", metrics)

    if isinstance(metrics, list):
        metrics_names = [metric.get('name', 'Unnamed Metric') for metric in metrics]
        metrics_scores = [metric.get('score', 0) for metric in metrics]

        # Create a bar chart
        fig, ax = plt.subplots()
        bar_width = 0.35  # Width of the bars
        x = range(len(metrics_names))

        # Create bars
        ax.bar(x, metrics_scores, width=bar_width, color=['skyblue', 'lightgreen', 'coral', 'gold', 'lightblue'])

        # Add labels and title
        ax.set_ylabel('Score')
        ax.set_title('Benchmark Results')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)  # Set y-axis limits (0 to 1)

        st.pyplot(fig)
    else:
        st.error("Metrics data is not in the expected format.")

# Main app logic
def main():
    st.title("Hugging Face Modell Scanner")
    st.write("Um verfügbare deutschsprachige LLMs auf Hugging Face zu finden, bitte den Scan starten.")

    # Initialize session state variables
    if 'models_with_popularity' not in st.session_state:
        st.session_state.models_with_popularity = None
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'benchmark_done' not in st.session_state:
        st.session_state.benchmark_done = False

    # Button to trigger the model scan
    if st.button('Modell Scan jetzt starten'):
        st.write("Modell Scan läuft...")
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
                "Bitte wählen Sie ein Modell aus:",
                model_options
            )

            # Extract model name from the selection
            selected_model_name = st.session_state.selected_model.split(" ")[0]

            # Display the benchmark button after model selection
            if st.button('Benchmark ausführen'):
                st.write("Benchmark wird ausgeführt...")

                # Load model and tokenizer
                model, tokenizer = load_model_and_tokenizer(selected_model_name)

                # Calculate BLEU score
                average_bleu_score, score_category = calculate_bleu(model_name=selected_model_name, max_index=10)

                # Calculate ROUGE score
                avg_rouge1, avg_rouge2, avg_rougeL = run_rouge(model_name=selected_model_name, tokenizer=tokenizer, max_index=10)

                # Create a metrics list for visualization
                metrics = [
                    {"name": "Average BLEU Score", "score": float(average_bleu_score)},
                    {"name": "Average ROUGE-1 Score", "score": float(avg_rouge1)},
                    {"name": "Average ROUGE-2 Score", "score": float(avg_rouge2)},
                    {"name": "Average ROUGE-L Score", "score": float(avg_rougeL)}
                ]

                # Display benchmark results
                display_benchmark_results(metrics)

                st.session_state.benchmark_done = True
        else:
            st.error("Fehler 01: Keine Modelle gefunden.")
    else:
        st.error("Bitte den Scan starten, um Modelle zu finden.")

if __name__ == "__main__":
    main()
