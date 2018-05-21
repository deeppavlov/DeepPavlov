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
from overrides import overrides

import nltk
import sys
import os
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.prints import RedirectedPrints
import string
import sentencepiece as spm

log = get_logger(__name__)


@register("bpe_tokenizer")
class BPETokenizer(Estimator):
    """
    Module uses sentencepiece to perform bpe tokenizing.
    See for details: https://github.com/google/sentencepiece/blob/master/python/README.md

    """

    def __init__(self, load_path, save_path=None, vocab_size=None, *args, **kwargs):
        self.super().__init__(load_path, save_path=None, *args, **kwargs)
        if self.load_path:
            self.model = self.load()
        elif vocab_size is None:
            log.error('No load_path provided but vocab_size is not defined. Provide vocab_size or load_path.')
            sys.exit(1)
        else:
            self.vocab_size = vocab_size
            if save_path is None:
                log.warning('No save_path provided, default save_path: ./bpe_model/')
                self.save_path = "./bpe_model"
                try:
                    os.mkdir(self.save_path)
                except OSError:
                    log.error('save_path is already exists')
        self._preprocess_table = str.maketrans({key: None for key in string.punctuation})

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load model from file
        """
        if self.load_path:
            log.info("Loading model {} from path: '{}'."
                     .format(self.__class__.__name__, self.load_path))
            model_file = str(self.load_path)
            try:
                model = spm.SentencePieceProcessor()
                model.Load(self.load_path)
            except Exception as e:
                log.error(e.__repr__())
                sys.exit(1)
        else:
            log.error('Provided load_path "{}" is incorrect.'
                      .format(self.load_path))
            sys.exit(1)
        return model

    def fit(self, train_file_path, *args, **kwargs):
        """
        Create vocab and model and save them on disk.

        :param train_file_path: path to the corpus file to obtain a vocabulary;
               lowercasing and punctuation removing should be performed on the corpus in advance.
        """
        with RedirectedPrints():
            try:
                spm.SentencePieceTrainer.Train('--input={} --model_prefix={}/bpe --vocab_size={}'.
                                           format(train_file_path, self.save_path, self.vocab_size))
            except Exception as e:
                log.error(e.__repr__())
                sys.exit(1)

        self.load_path = self.save_path + 'bpe.model'
        self.model = self.load()

    @overrides
    def __call__(self, batch, preprocess=True, *args, **kwargs):
        """
        tokenize batch
        """
        encoded = []
        if preprocess:
            for s in self._preprocess(batch):
                encoded.append(self.model.EncodeAsPieces(s))
        else:
            for s in batch:
                encoded.append(self.model.EncodeAsPieces(s))
        return encoded

    def _preprocess(self, batch):
        """
        turning strings into lowercase and remove punctuation
        :param batch: list of strings
        :yield: string
        """
        for s in batch:
            yield s.lower().s.translate(self._preprocess_table)
