import numpy as np
from deeppavlov.core.common.registry import register


@register('emb_mat_assembler')
class EmbeddingsMatrixAssembler:
    def __init__(self, embedder, vocab=None):
        self.emb_mat = np.zeros([len(vocab), embedder.emb_dim], dtype=np.float32)
        if vocab is not None:
            tokens = vocab.tokens
        else:
            tokens = embedder.tokens
        for n, token in enumerate(tokens):
            self.emb_mat[n] = embedder(token)