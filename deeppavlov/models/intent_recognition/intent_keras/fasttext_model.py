
from deeppavlov.core.models.embedding_trainable import EmbeddingTrainableModel

from deeppavlov.core.models.embedding_inferable import EmbeddingInferableModel
from gensim.models.fasttext import FastText


fasttext_model = EmbeddingTrainableModel(embedding_dim=10)
print(fasttext_model.model)
f = open('/home/dilyara/data/data_files/embeddings/badwords.txt', 'r')
data = f.readlines()
fasttext_model.train(data)
print(fasttext_model.model)

fasttext_model.save(fname="/home/dilyara/data/data_files/embeddings/badwords_emb")
print(fasttext_model.model)
print('Done!')
print(fasttext_model.infer(['fuck']))


fasttext_model = EmbeddingInferableModel(fname="/home/dilyara/data/data_files/embeddings/badwords_emb.bin",
                                         embedding_dim=10,
                                         emb_dict_name="/home/dilyara/data/data_files/embeddings/badwords_emb.emb")
print(fasttext_model.model)


print(fasttext_model.model)
print('Done!')
print(fasttext_model.infer(['.']))
