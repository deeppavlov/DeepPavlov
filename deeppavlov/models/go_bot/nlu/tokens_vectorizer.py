from logging import getLogger
from typing import List, Optional

import numpy as np

log = getLogger(__name__)


# todo logging
class TokensVectorRepresentationParams:
    """the DTO-like class to transfer TokenVectorizer's vectorizers dimensions"""

    def __init__(self, embedding_dim: Optional[int], bow_dim: Optional[int]):
        self.embedding_dim = embedding_dim
        self.bow_dim = bow_dim


class TokensVectorizer:
    """
    the TokensVectorizer class is used in the NLU part of deeppavlov go-bot pipeline.
    (for more info on NLU logic see the NLUManager --- the go-bot NLU main class)

    TokensVectorizer is manages the BOW tokens encoding and tokens embedding.
    Both BOW encoder and embedder are optional and have to be pre-trained:
    this class wraps their usage but not training.
    """

    def __init__(self, debug, word_vocab=None, bow_embedder=None, embedder=None):
        # todo adequate type hints
        self.debug = debug
        self.word_vocab = word_vocab  # TODO: isn't it passed with bow embedder?
        self.bow_embedder = bow_embedder
        self.embedder = embedder

    def _use_bow_encoder(self) -> bool:
        """
        Returns:
            is BOW encoding enabled in the TokensVectorizer
        """
        return callable(self.bow_embedder)

    def _embed_tokens(self, tokens: List[str], mean_embeddings: bool) -> Optional[np.ndarray]:
        """
        Args:
            tokens: list of tokens to embed
            mean_embeddings: if True, will return the mean vector of calculated embeddings sequence.
                             otherwise will return the calculated embeddings sequence.

        Returns:
            the (maybe averaged vector of) calculated embeddings sequence and None if embedder is disabled.
        """
        tokens_embedded = np.array([], dtype=np.float32)
        if callable(self.embedder):
            tokens_embedded = self.embedder([tokens], mean=mean_embeddings)[0]
        return tokens_embedded

    def bow_encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Args:
            tokens: list of tokens to BOW encode

        Returns:
            if uses BOW encoder, returns np array with BOW encoding for tokens.
            Otherwise returns an empty list.
        """
        bow_features = np.array([], dtype=np.float32)
        if self._use_bow_encoder():
            tokens_idx = self.word_vocab(tokens)
            bow_features = self.bow_embedder([tokens_idx])[0]
            bow_features = bow_features.astype(np.float32)
        return bow_features

    @staticmethod
    def _standard_normal_like(source_vector: np.ndarray) -> np.ndarray:
        """
        Args:
            source_vector: the vector of which to follow the result shape

        Returns:
            the standard normal distribution of the shape of the source vector
        """
        vector_dim = source_vector.shape[0]
        return np.random.normal(loc=0.0, scale=1 / vector_dim, size=vector_dim)

    @staticmethod
    def _pad_sequence_to_size(out_sequence_length: int, token_dim: int, tokens_embedded: np.ndarray) -> np.ndarray:
        """
        Pad the passed vectors sequence to the specified length.

        Args:
            out_sequence_length: the length to pad sequence to
            token_dim: the shape of output embedding
            tokens_embedded: some sequence of vectors

        Returns:
            the padded sequence of vectors
        """
        out_sequence_length = out_sequence_length - len(tokens_embedded)
        padding = np.zeros(shape=(out_sequence_length, token_dim), dtype=np.float32)
        if tokens_embedded:
            emb_context = np.concatenate((padding, np.array(tokens_embedded)))
        else:
            emb_context = padding
        return emb_context

    def calc_tokens_mean_embedding(self, tokens: List[str]) -> np.ndarray:
        """
        Args:
            tokens: list of tokens to embed

        Returns:
            the average vector of embeddings sequence
            or if avg is zeros then the standard normal distributed random vector instead.
            None if embedder is disabled.
        """
        tokens_embedded = self._embed_tokens(tokens, True)
        # random embedding instead of zeros
        if tokens_embedded.size != 0 and np.all(tokens_embedded < 1e-20):
            # TODO:  size != 0 not pythonic
            tokens_embedded = np.fabs(self._standard_normal_like(tokens_embedded))
        return tokens_embedded

    def calc_tokens_embeddings(self, output_sequence_length: int, token_dim: int, tokens: List[str]) -> np.ndarray:
        """
        Calculate embeddings of passed tokens.
        Args:
            output_sequence_length: the length of sequence to output
            token_dim: the shape of output embedding
            tokens: list of tokens to embed

        Returns:
            the padded sequence of calculated embeddings
        """
        tokens_embedded = self._embed_tokens(tokens, False)
        if tokens_embedded is not None:
            emb_context = self._pad_sequence_to_size(output_sequence_length, token_dim, tokens_embedded)
        else:
            emb_context = np.array([], dtype=np.float32)
        return emb_context

    def get_dims(self) -> TokensVectorRepresentationParams:
        """
        Returns:
            the TokensVectorRepresentationParams with embedder and BOW encoder output dimensions.
            None instead of the missing dim if BOW encoder or embedder are missing.
        """
        embedder_dim = self.embedder.dim if self.embedder else None
        bow_encoder_dim = len(self.word_vocab) if self.bow_embedder else None
        return TokensVectorRepresentationParams(embedder_dim, bow_encoder_dim)
