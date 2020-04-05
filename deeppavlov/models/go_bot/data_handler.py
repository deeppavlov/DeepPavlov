import re
from logging import getLogger
from typing import List

import numpy as np

log = getLogger(__name__)

class TokensVectorRepresentationParams:
    def __init__(self, embedding_dim, bow_dim):
        self.embedding_dim = embedding_dim
        self.bow_dim = bow_dim

class TokensVectorizer:

    def __init__(self, debug, word_vocab, bow_embedder, embedder):
        self.debug = debug
        self.word_vocab = word_vocab
        self.bow_embedder = bow_embedder
        self.embedder = embedder

    def use_bow_encoder(self):
        return callable(self.bow_embedder)

    def word_vocab_size(self):
        return len(self.word_vocab) if self.word_vocab else None

    def encode_tokens(self, tokens: List[str], mean_embeddings):

        bow_features = self.bow_encode_tokens(tokens)
        tokens_embedded = self.embed_tokens(tokens, mean_embeddings)

        return bow_features, tokens_embedded

    def embed_tokens(self, tokens, mean_embeddings):
        tokens_embedded = None  # todo worst name ever
        if callable(self.embedder):
            tokens_embedded = self.embedder([tokens], mean=mean_embeddings)[0]
        return tokens_embedded

    def bow_encode_tokens(self, tokens):
        bow_features = []
        if self.use_bow_encoder():
            tokens_idx = self.word_vocab(tokens)
            bow_features = self.bow_embedder([tokens_idx])[0]
            bow_features = bow_features.astype(np.float32)
        return bow_features

    @staticmethod
    def standard_normal_like(source_vector):
        vector_dim = source_vector.shape[0]
        return np.random.normal(0, 1 / vector_dim, vector_dim)

    @staticmethod
    def pad_sequence_to_size(padding_length, token_dim, tokens_embedded):
        padding_length = padding_length - len(tokens_embedded)
        padding = np.zeros(shape=(padding_length, token_dim), dtype=np.float32)
        if tokens_embedded:
            emb_context = np.concatenate((padding, np.array(tokens_embedded)))
        else:
            emb_context = padding
        return emb_context

    def calc_tokens_embedding(self, tokens):
        emb_features = self.embed_tokens(tokens, True)
        # random embedding instead of zeros
        if np.all(emb_features < 1e-20):
            emb_features = np.fabs(self.standard_normal_like(emb_features))
        return emb_features

    def calc_tokens_embeddings(self, padding_length, token_dim, tokens):
        tokens_embedded = self.embed_tokens(tokens, False)
        if tokens_embedded is not None:
            emb_context = self.pad_sequence_to_size(padding_length, token_dim, tokens_embedded)
        else:
            emb_context = np.array([], dtype=np.float32)
        return emb_context

    def get_dims(self):
        embedder_dim = self.embedder.dim if self.embedder else None
        bow_encoder_dim = len(self.word_vocab) if self.bow_embedder else None
        return TokensVectorRepresentationParams(embedder_dim, bow_encoder_dim)