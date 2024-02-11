from sentence_transformers import SentenceTransformer
import torch

# Choose and initialize model here
model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_semantic_similarity(text1, text2):
    # Encode the sentences
    embeddings = model.encode([text1, text2])

    # Calculate cosine similarity of embeddings
    similarity = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0)
    )
    return similarity.item()
