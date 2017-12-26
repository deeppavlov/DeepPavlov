import numpy as np

from deeppavlov.core.models.fasttext_inferable import EmbeddingInferableModel

fasttext_model = EmbeddingInferableModel(embedding_fname="/home/dilyara/data/data_files/embeddings/reddit_fasttext_model.bin",
                                         embedding_dim=100)
print(fasttext_model.model)

print('Done!')

embed = fasttext_model.infer('man woman boy girl')
a = embed[0] - embed[1]
b = embed[2] - embed[3]
print(a)
print(b)
print(np.linalg.norm(a - b) / np.linalg.norm(a))
print(np.linalg.norm(a - b) / np.linalg.norm(b))
