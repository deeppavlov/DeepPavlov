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

import numpy as np
from typing import Type

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.data.vocab import DefaultVocabulary
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.encoders.bow import BoWEncoder
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer
from deeppavlov.skills.seq2seq_go_bot.network import Seq2SeqGoalOrientedBotNetwork
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("seq2seq_go_bot")
class Seq2SeqGoalOrientedBot(NNModel):
    def __init__(self,
                 tokenizer: Type = StreamSpacyTokenizer,
                 network: Type = Seq2SeqGoalOrientedBotNetwork,
                 source_vocab: Type = DefaultVocabulary,
                 target_vocab: Type = DefaultVocabulary,
                 bow_encoder=None,
                 debug=False,
                 save_path=None,
                 **kwargs):

        super().__init__(save_path=save_pathm, mode=kwargs['mode'])

        self.tokenizer = tokenizer
        self.network = network
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.bow_encoder = bow_encoder
        self.embedder = embedder
        self.debug = debug

    def train_on_batch(self, x, y):
        b_features, b_outs = [], []
        for d_contexts, d_responses in zip(x, y):

            #b_features.append([])
            #b_outs.append([])

            for context, response in zip(d_contexts, d_responses):
                features = self._encode_context(context['text'])
                outs = self._encode_response(response['text'])

                b_features.append(features)
                b_outs.append(outs)

        self.network.train_on_batch(b_features, b_outs)

    def __call__(self, batch):
        if isinstance(batch[0], str):
            return [self._infer(x) for x in batch]
        return [self._infer_dialog(x) for x in batch]

    def _infer(self, context):
        pass

    def _infer_dialog(self, contexts):
        for context in contexts:
            pass

    def save(self):
        self.network.save()

    def load(self):
        pass
