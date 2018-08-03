import numpy as np
from deeppavlov.core.common.registry import register
from sklearn.decomposition import PCA


@register('emb_mat_assembler')
class EmbeddingsMatrixAssembler:
    """Assembles matrix of embeddings obtained from some embedder."""
    def __init__(self, embedder, vocab, character_level=False, emb_dim=None, estimate_by_n=10000, *args, **kwargs):
        if emb_dim is None:
            emb_dim = embedder.dim
        self.emb_mat = np.zeros([len(vocab), emb_dim], dtype=np.float32)
        tokens_for_estimation = list(embedder)[:estimate_by_n]
        estimation_matrix = np.array([embedder([[word]])[0][0] for word in tokens_for_estimation], dtype=np.float32)
        emb_std = np.std(estimation_matrix)

        if emb_dim < embedder.dim:
            pca = PCA(n_components=emb_dim)
            pca.fit(estimation_matrix)
        elif emb_dim > embedder.dim:
            raise RuntimeError(f'Model dimension must be greater then requsted embeddings '
                               'dimension! model_dim = {embedder.dim}, requested_dim = {emb_dim}')
        else:
            pca = None
        for n, token in enumerate(vocab):
            if character_level:
                char_in_word_bool = np.array([token in word for word in tokens_for_estimation], dtype=bool)
                all_words_with_character = estimation_matrix[char_in_word_bool]
                if len(all_words_with_character) != 0:
                    if pca is not None:
                        all_words_with_character = pca.transform(all_words_with_character)
                    self.emb_mat[n] = sum(all_words_with_character) / len(all_words_with_character)
                else:
                    self.emb_mat[n] = np.random.randn(emb_dim) * np.std(self.emb_mat[:n])
            else:
                try:
                    if pca is not None:
                        self.emb_mat[n] = pca(embedder([[token]])[0])[0]
                    else:
                        self.emb_mat[n] = embedder([[token]])[0][0]

                except KeyError:
                    self.emb_mat[n] = np.random.randn(emb_dim) * emb_std

    @property
    def dim(self):
        return self.emb_mat.shape[1]


@register('random_emb_mat')
class RandomEmbeddingsMatrix:
    """Assembles matrix of random embeddings."""
    def __init__(self, vocab_len, emb_dim, *args, **kwargs):
        self.emb_mat = np.random.randn(vocab_len, emb_dim).astype(np.float32) / np.sqrt(emb_dim)
