from transformers import BertTokenizer, BertModel
import torch

# Initialize BERT-Model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def calculate_semantic_similarity(text1, text2):
    embedding1 = get_bert_embeddings(text1)
    embedding2 = get_bert_embeddings(text2)
    # Calculate cosine similarity of embeddings
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return similarity.item()
