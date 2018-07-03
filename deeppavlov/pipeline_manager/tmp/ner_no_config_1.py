from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_iterators.dstc2_ner_iterator import Dstc2NerDatasetIterator

from deeppavlov.models.ner.network import NerNetwork
from deeppavlov.models.preprocessors.mask import Mask
from deeppavlov.models.preprocessors.lazy_tokenizer import LazyTokenizer
from deeppavlov.models.preprocessors.str_lower import StrLower
from deeppavlov.models.preprocessors.assemble_embeddins_matrix import RandomEmbeddingsMatrix


set_deeppavlov_root({})
data_path = expand_path('dstc2')

ds = Dstc2NerDatasetIterator(DSTC2DatasetReader().read(data_path=data_path), dataset_path=data_path)

metrics = ['f1_macro']

ner_params = {"token_emb_mat": "#random_emb.emb_mat",
              "n_hidden_list": [64, 64],
              "net_type": "cnn",
              "n_tags": '#tag_vocab.len',
              "save_path": "slotfill_dstc2/model",
              "load_path": "slotfill_dstc2/model",
              "embeddings_dropout": True,
              "top_dropout": True,
              "intra_layer_dropout": False,
              "use_batch_norm": True,
              "learning_rate": 1e-2,
              "dropout_keep_prob": 0.5}

train_config = {'batch_size': 64,
                'metric_optimization': 'maximize',

                'validation_patience': 5,
                'val_every_n_epochs': 3,

                'log_every_n_batches': 0,
                'log_every_n_epochs': 1,

                'validate_best': True,
                'test_best': True}

pipe = [LazyTokenizer(in_x="x", name="lazy_tokenizer", out_params="x_tok"),
        StrLower(in_x="x_tok", name="str_lower", out_params="x_lower"),
        SimpleVocabulary(in_x="x_tok", id="word_vocab", name="simple_vocab", pad_with_zeros=True, fit_on="x_tok",
                         save_path="slotfill_dstc2/word.dict", load_path="slotfill_dstc2/word.dict",
                         out_params="x_tok_ind"),
        SimpleVocabulary(in_x="y", id="tag_vocab", name="simple_vocab", pad_with_zeros=True, fit_on="y",
                         save_path="slotfill_dstc2/tag.dict", load_path="slotfill_dstc2/tag.dict",
                         out_params="y_ind"),
        (RandomEmbeddingsMatrix, {'vocab_len': '#word_vocab.len', 'emb_dim': 100}),
        Mask(in_x="x", name="lazy_tokenizer", out_params="x_tok"),
        (NerNetwork, ner_params),
        {"ref": "tag_vocab", "in": ["y_predicted"], "out": ["tags"]}]

#####################################################################################################################
tok = LazyTokenizer(in_x="x", name="lazy_tokenizer", out_params="x_tok")
lower = StrLower(in_x="x_tok", name="str_lower", out_params="x_lower")
word_vocab = SimpleVocabulary(in_x="x_tok", name="word_vocab", pad_with_zeros=True, fit_on="x_tok",
                              save_path="slotfill_dstc2/word.dict", load_path="slotfill_dstc2/word.dict",
                              out_params="x_tok_ind")
tag_vocab = SimpleVocabulary(in_x="y", name="tag_vocab", pad_with_zeros=True, fit_on="y",
                             save_path="slotfill_dstc2/tag.dict", load_path="slotfill_dstc2/tag.dict",
                             out_params="y_ind")
mask = Mask(in_x="x", name="lazy_tokenizer", out_params="x_tok")
emb = RandomEmbeddingsMatrix(emb_dim=100, vocab_len=100)
ner_network = NerNetwork(**ner_params)

pipe_ = [tok,
         lower,
         word_vocab,
         tag_vocab,
         (emb, {'ref_op_name': 'attribute_name'}),
         mask,
         (ner_network, {'ref_op_name': 'attribute_name'}),
         (tag_vocab, {"in": ["y_predicted"], "out": ["tags"]})]
#####################################################################################################################

pipe_1 = [(LazyTokenizer, {"in_x": "x", "name": "lazy_tokenizer", "out_params": "x_tok"}),
          (StrLower, {"in_x": "x_tok", "name": "str_lower", "out_params": "x_lower"}),
          (SimpleVocabulary, {"in_x": "x_tok", "id": "word_vocab", "name": "simple_vocab", "pad_with_zeros": True,
                              "fit_on": "x_tok", "save_path": "slotfill_dstc2/word.dict",
                              "load_path": "slotfill_dstc2/word.dict", "out_params": "x_tok_ind"}),
          (SimpleVocabulary, {"in_x": "y", "id": "tag_vocab", "name": "simple_vocab", "pad_with_zeros": True,
                              "fit_on": "y", "save_path": "slotfill_dstc2/tag.dict",
                              "load_path": "slotfill_dstc2/tag.dict", "out_params": "y_ind"}),
          (RandomEmbeddingsMatrix, {'vocab_len': '#word_vocab.len', 'emb_dim': 100}),
          (Mask, {"in_x": "x", "name": "lazy_tokenizer", "out_params": "x_tok"}),
          (NerNetwork, ner_params),
          {"ref": "tag_vocab", "in": ["y_predicted"], "out": ["tags"]}]

pipeline = Pipeline(pipe, ds, train_config, metrics)
