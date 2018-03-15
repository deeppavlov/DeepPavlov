# check version
import sys
assert sys.hexversion >= 0x3060000, 'Does not work in python3.5 or lower'


import deeppavlov.core.models.keras_model
import deeppavlov.core.data.dataset_iterator
import deeppavlov.core.data.vocab
import deeppavlov.dataset_readers.babi_reader
import deeppavlov.dataset_readers.dstc2_reader
import deeppavlov.dataset_readers.conll2003_reader
import deeppavlov.dataset_readers.typos_reader
import deeppavlov.dataset_readers.csv_classification_reader
import deeppavlov.dataset_iterators.dialog_iterator
import deeppavlov.dataset_iterators.dstc2_ner_iterator
import deeppavlov.dataset_iterators.dstc2_intents_iterator
import deeppavlov.dataset_iterators.typos_iterator
import deeppavlov.dataset_iterators.basic_classification_iterator
import deeppavlov.models.classifiers.intents.intent_model
import deeppavlov.models.commutators.random_commutator
import deeppavlov.models.embedders.fasttext_embedder
import deeppavlov.models.embedders.dict_embedder
import deeppavlov.models.encoders.bow
import deeppavlov.models.ner.slotfill
import deeppavlov.models.spellers.error_model.error_model
import deeppavlov.models.trackers.hcn_at
import deeppavlov.models.trackers.hcn_et
import deeppavlov.models.preprocessors.str_lower
import deeppavlov.models.ner.ner
import deeppavlov.skills.go_bot.go_bot
import deeppavlov.skills.go_bot.network
import deeppavlov.skills.go_bot.tracker
import deeppavlov.vocabs.typos

import deeppavlov.metrics.accuracy
import deeppavlov.metrics.fmeasure

import deeppavlov.core.common.log
