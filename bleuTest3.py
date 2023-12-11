#diese bleuTest wird in separaten Task aktualisiert.
# import json
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#
#
# def run_bleu_test_on_json_dataset(json_file_path):
#     with open(json_file_path, 'r', encoding='utf-8') as file:
#         dataset = json.load(file)
#
#     total_bleu_score = 0
#     smoothing = SmoothingFunction()
#
#     for i, data_point in enumerate(dataset):
#         predictions = data_point.get("predictions", [])
#         references = data_point.get("references", [])
#
#         if not predictions or not references:
#             print(f"Fehlerhafter Datenpunkt {i + 1}: Fehlende Vorhersagen oder Referenzen.")
#             continue
#
#         for j, prediction in enumerate(predictions):
#             if j >= len(references):
#                 print(f"Fehlerhafter Datenpunkt {i + 1}: Index {j} außerhalb des gültigen Bereichs für Referenzen.")
#                 continue
#
#             reference_set = references[j]
#             bleu_score = sentence_bleu(reference_set, prediction, smoothing_function=smoothing.method1)
#             total_bleu_score += bleu_score
#
#             print(f"BLEU Score for Prediction {i + 1}-{j + 1}: {bleu_score}")
#
#     average_bleu_score = total_bleu_score / len(dataset)
#     print(f"\nAverage BLEU Score for the Entire Dataset: {average_bleu_score}")
#
#
# # Beispielaufruf:
# json_file_path = "dataset/bleu_dataset.json"
# run_bleu_test_on_json_dataset(json_file_path)
