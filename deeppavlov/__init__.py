# check version
import sys
assert sys.hexversion >= 0x3060000, 'Does not work in python3.5 or lower'


import deeppavlov.core.models.keras_model
import deeppavlov.core.data.vocab
import deeppavlov.dataset_readers.babi_reader
import deeppavlov.dataset_readers.dstc2_reader
import deeppavlov.dataset_readers.kvret_reader
import deeppavlov.dataset_readers.conll2003_reader
import deeppavlov.dataset_readers.typos_reader
import deeppavlov.dataset_readers.basic_classification_reader
import deeppavlov.dataset_readers.squad_dataset_reader
import deeppavlov.dataset_iterators.dialog_iterator
import deeppavlov.dataset_iterators.kvret_dialog_iterator
import deeppavlov.dataset_iterators.dstc2_ner_iterator
import deeppavlov.dataset_iterators.dstc2_intents_iterator
import deeppavlov.dataset_iterators.typos_iterator
import deeppavlov.dataset_iterators.basic_classification_iterator
import deeppavlov.dataset_iterators.squad_iterator
import deeppavlov.dataset_iterators.sqlite_iterator
import deeppavlov.models.classifiers.intents.intent_model
import deeppavlov.models.commutators.random_commutator
import deeppavlov.models.embedders.fasttext_embedder
import deeppavlov.models.embedders.dict_embedder
import deeppavlov.models.embedders.glove_embedder
import deeppavlov.models.embedders.bow_embedder
import deeppavlov.models.ner.slotfill
import deeppavlov.models.ner.ner
import deeppavlov.models.ner.ner_ontonotes
import deeppavlov.models.spellers.error_model.error_model
import deeppavlov.models.trackers.hcn_at
import deeppavlov.models.trackers.hcn_et
import deeppavlov.models.preprocessors.str_lower
import deeppavlov.models.preprocessors.squad_preprocessor
import deeppavlov.models.ner.ner
import deeppavlov.models.tokenizers.spacy_tokenizer
import deeppavlov.models.tokenizers.split_tokenizer
import deeppavlov.models.tokenizers.ru_tokenizer
import deeppavlov.models.squad.squad
import deeppavlov.skills.go_bot.bot
import deeppavlov.skills.go_bot.network
import deeppavlov.skills.go_bot.tracker
import deeppavlov.skills.seq2seq_go_bot.bot
import deeppavlov.skills.seq2seq_go_bot.network
import deeppavlov.skills.seq2seq_go_bot.kb
import deeppavlov.skills.odqa.ranker
import deeppavlov.vocabs.typos
import deeppavlov.vocabs.wiki_sqlite
import deeppavlov.dataset_readers.insurance_reader
import deeppavlov.dataset_iterators.ranking_iterator
import deeppavlov.models.ranking.ranking_model
import deeppavlov.models.ranking.metrics

import deeppavlov.metrics.accuracy
import deeppavlov.metrics.fmeasure
import deeppavlov.metrics.bleu
import deeppavlov.metrics.squad_metrics

import deeppavlov.core.common.log
