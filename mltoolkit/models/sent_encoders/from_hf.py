import torch
#import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

def from_hf(
        model_name: str,
        emb_dim: int, 
        max_seq_len: int,
        pooling_mode: str='mean',
        cache_dir: str=None) -> SentenceTransformer:
    """
    builds a SentenceTransformer model from a huggingface model name and other args

    :param model_name:
    :type model_name: str

    :param emb_dim: the dimensionality of the embeddings
    :type emb_dim: int

    :param max_seq_len: the maximum sequence length of tokens that the model will encode
    :type max_seq_len: int

    :pooling_mode: the method for pooling the token embeddings into 1 sentence embedding
    :type pooling_mode: str

    :param cache_dir: the directory to cache the hf model
    :type cache_dir: str

    :return: a SentenceTransformer model
    :rtype: SentenceTransformer
    """

    base_model = Transformer(
        model_name,
        cache_dir=cache_dir,
        max_seq_length=max_seq_len,
        tokenizer_args={'padding': True}
    )

    pooler = Pooling(
        emb_dim,
        pooling_mode=pooling_mode,
    )

    model = SentenceTransformer(
        modules=[base_model, pooler],
        cache_folder=cache_dir,
        trust_remote_code=True
    )

    return model