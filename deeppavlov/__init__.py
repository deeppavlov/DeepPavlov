"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from .package_meta import __version__, __author__

# check version
import sys
assert sys.hexversion >= 0x3060000, 'Does not work in python3.5 or lower'

import deeppavlov.core.models.keras_model
import deeppavlov.core.data.vocab
import deeppavlov.core.data.simple_vocab
import deeppavlov.core.data.sqlite_database
import deeppavlov.dataset_readers.babi_reader
import deeppavlov.dataset_readers.dstc2_reader
import deeppavlov.dataset_readers.kvret_reader
import deeppavlov.dataset_readers.conll2003_reader
import deeppavlov.dataset_readers.typos_reader
import deeppavlov.dataset_readers.basic_classification_reader
import deeppavlov.dataset_readers.squad_dataset_reader
import deeppavlov.dataset_readers.morphotagging_dataset_reader

import deeppavlov.dataset_iterators.dialog_iterator
import deeppavlov.dataset_iterators.kvret_dialog_iterator
import deeppavlov.dataset_iterators.dstc2_ner_iterator
import deeppavlov.dataset_iterators.dstc2_intents_iterator
import deeppavlov.dataset_iterators.typos_iterator
import deeppavlov.dataset_iterators.basic_classification_iterator
import deeppavlov.dataset_iterators.squad_iterator
import deeppavlov.dataset_iterators.sqlite_iterator
import deeppavlov.dataset_iterators.morphotagger_iterator

import deeppavlov.models.go_bot.bot
import deeppavlov.models.go_bot.network
import deeppavlov.models.go_bot.tracker
import deeppavlov.models.seq2seq_go_bot.bot
import deeppavlov.models.seq2seq_go_bot.network
import deeppavlov.models.seq2seq_go_bot.kb
import deeppavlov.models.classifiers.intents.intent_model
import deeppavlov.models.commutators.random_commutator
import deeppavlov.models.embedders.fasttext_embedder
import deeppavlov.models.embedders.dict_embedder
import deeppavlov.models.embedders.glove_embedder
import deeppavlov.models.embedders.bow_embedder
import deeppavlov.models.spelling_correction.brillmoore.error_model
import deeppavlov.models.spelling_correction.levenstein.searcher_component
import deeppavlov.models.spelling_correction.electors.kenlm_elector
import deeppavlov.models.spelling_correction.electors.top1_elector
import deeppavlov.models.trackers.hcn_at
import deeppavlov.models.trackers.hcn_et
import deeppavlov.models.preprocessors.str_lower
import deeppavlov.models.preprocessors.squad_preprocessor
import deeppavlov.models.preprocessors.capitalization
import deeppavlov.models.preprocessors.dirty_comments_preprocessor
import deeppavlov.models.tokenizers.nltk_tokenizer
import deeppavlov.models.tokenizers.nltk_moses_tokenizer
import deeppavlov.models.tokenizers.spacy_tokenizer
import deeppavlov.models.tokenizers.split_tokenizer
import deeppavlov.models.tokenizers.ru_tokenizer
import deeppavlov.models.squad.squad
import deeppavlov.models.morpho_tagger.tagger
import deeppavlov.models.morpho_tagger.common
import deeppavlov.models.api_requester

import deeppavlov.skills.odqa.tfidf_ranker
import deeppavlov.vocabs.typos
import deeppavlov.vocabs.wiki_sqlite
import deeppavlov.dataset_readers.insurance_reader
import deeppavlov.dataset_iterators.ranking_iterator
import deeppavlov.models.ner.network
import deeppavlov.models.ranking.ranking_model
import deeppavlov.models.ranking.metrics
import deeppavlov.models.preprocessors.char_splitter
import deeppavlov.models.preprocessors.mask
import deeppavlov.models.preprocessors.assemble_embeddins_matrix
import deeppavlov.models.preprocessors.capitalization
import deeppavlov.models.preprocessors.field_getter
import deeppavlov.models.preprocessors.sanitizer
import deeppavlov.models.preprocessors.lazy_tokenizer
import deeppavlov.models.slotfill.slotfill_raw
import deeppavlov.models.slotfill.slotfill
import deeppavlov.models.preprocessors.one_hotter
import deeppavlov.dataset_readers.ontonotes_reader

import deeppavlov.models.classifiers.tokens_matcher.tokens_matcher


import deeppavlov.metrics.accuracy
import deeppavlov.metrics.fmeasure
import deeppavlov.metrics.bleu
import deeppavlov.metrics.squad_metrics
import deeppavlov.metrics.roc_auc_score
import deeppavlov.metrics.fmeasure_classification

import deeppavlov.core.common.log

import deeppavlov.download
