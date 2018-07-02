from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_iterators.dstc2_ner_iterator import Dstc2NerDatasetIterator

from deeppavlov.models.ner.network import NerNetwork
from deeppavlov.models.preprocessors.mask import Mask
from deeppavlov.models.preprocessors.lazy_tokenizer import LazyTokenizer
from deeppavlov.models.preprocessors.str_lower import StrLower

from deeppavlov.metrics.accuracy import per_item_accuracy
from deeppavlov.metrics.fmeasure import ner_f1
from deeppavlov.models.preprocessors.assemble_embeddins_matrix import RandomEmbeddingsMatrix

from deeppavlov.core.commands.train import _train_batches

set_deeppavlov_root({})
data_path = expand_path('dstc2')

ds = Dstc2NerDatasetIterator(DSTC2DatasetReader().read(data_path=data_path), dataset_path=data_path)

chainer = Chainer(in_y='y')

x, y = ds.get_instances('train')

word_vocab = SimpleVocabulary(in_x="x_tok",
                              id="word_vocab",
                              name="simple_vocab",
                              pad_with_zeros=True,
                              fit_on="x_tok",
                              save_path="slotfill_dstc2/word.dict",
                              load_path="slotfill_dstc2/word.dict",
                              out_params="x_tok_ind")

tag_vocab = SimpleVocabulary(in_x="y",
                             id="tag_vocab",
                             name="simple_vocab",
                             pad_with_zeros=True,
                             fit_on="y",
                             save_path="slotfill_dstc2/tag.dict",
                             load_path="slotfill_dstc2/tag.dict",
                             out_params="y_ind")

# print("here ---->", tag_vocab.in_x)

word_vocab.fit(x)
tag_vocab.fit(x)

lower = StrLower(in_x="x", name="str_lower", out_params="x_lower")

random_emb = RandomEmbeddingsMatrix(vocab_len=word_vocab.__len__(), emb_dim=100)

ner_params = {"token_emb_mat": random_emb.emb_mat,
              "n_hidden_list": [64, 64],
              "net_type": "cnn",
              "n_tags": tag_vocab.__len__(),
              "save_path": "slotfill_dstc2/model",
              "load_path": "slotfill_dstc2/model",
              "embeddings_dropout": True,
              "top_dropout": True,
              "intra_layer_dropout": False,
              "use_batch_norm": True,
              "learning_rate": 1e-2,
              "dropout_keep_prob": 0.5}

model = NerNetwork(**ner_params)
tokenizer = LazyTokenizer(in_x="x", name="lazy_tokenizer", out_params="x")
mask = Mask()

# chainer create
chainer.append(tokenizer, in_x='x', out_params=['x_tok'])
chainer.append(lower, in_x='x_tok', out_params='x_lower')
chainer.append(word_vocab, in_x='x_lower', out_params=['x_tok_ind'])
chainer.append(tag_vocab, in_x='y', out_params=['y_ind'])
chainer.append(mask, in_x='x_tok', out_params=['mask'])
chainer.append(model, in_x=['x_tok_ind', 'mask'], in_y='y_ind', main=True, out_params=['model_out'])
chainer.append(tag_vocab, in_x='model_out', out_params=['out'])

train_config = {'batch_size': 64,
                'metric_optimization': 'maximize',

                'validation_patience': 5,
                'val_every_n_epochs': 3,

                'log_every_n_batches': 0,
                'log_every_n_epochs': 1,

                'validate_best': True,
                'test_best': True}

_train_batches(chainer, ds, train_config, [('ner_f1', ner_f1), ('accuracy', per_item_accuracy)])
chainer.load()

x, y = ds.get_instances('valid')
print('valid', ner_f1(y, chainer(x)))

x, y = ds.get_instances('test')
print('test', ner_f1(y, chainer(x)))
