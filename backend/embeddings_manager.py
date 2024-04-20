from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import torch

# dict : huggingface url -> max token length (will be chunk size)
MODELS_DICT = {"intfloat/multilingual-e5-large": 512,
               "intfloat/multilingual-e5-large-instruct": 512}


def get_embedding_model(name: str):
    if name in MODELS_DICT:
        return HuggingFaceEmbeddings(model_name=name, model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {})
    else:
        raise ValueError(f"Model {name} not found in the list of available models")