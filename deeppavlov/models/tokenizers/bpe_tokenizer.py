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

import sys
import os
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.prints import RedirectedPrints
import string
from typing import List, Union
import sentencepiece as spm

log = get_logger(__name__)


@register("bpe_tokenizer")
class BPETokenizer(Estimator):
    """
    Module uses sentencepiece to perform bpe tokenizing.
    See for details: https://github.com/google/sentencepiece/

    """

    def __init__(self, load_path: str = None, save_path: str = None, vocab_size: int = None,
                 indexes_only=False, preprocess_call=True, pad_with_empty_str = False, *args, **kwargs):
        """
        :param load_path: path to saved model to load it from; if provided model will be initialized from this save.
        :param save_path: path for saving trained model if you would like to train a new one.
        :param vocab_size: provide vocab_size if you would like to train BPE model;
                           if you would like to load existing model you should't provide this parameter.
        :param indexes_only: if True works like a vocab; encoding into a sequence of indices
                             and decoding also from a sequence of indices.
        :param preprocess_call: if True

        """
        super().__init__(load_path=load_path, save_path=save_path, *args, **kwargs)

        self.indexes_only = indexes_only
        self.preprocess_call = preprocess_call
        self.padding = pad_with_empty_str

        if self.load_path:
            self.model = self.load()
            self._set_encoding_type()

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
                    log.warning('save_path: {} is already exists, using existing one.'.format(self.save_path))
        self._preprocess_table = str.maketrans({key: None for key in string.punctuation})

    def _set_encoding_type(self):
        if self.model:
            if self.indexes_only:
                log.info("Set encoding and decoding using indices only.")
                self._encode = self.model.EncodeAsIds
                self._decode = self.model.DecodeIds
            else:
                log.info("Set encoding and decoding using tokens only.")
                self._encode = self.model.EncodeAsPieces
                self._decode = self.model.DecodePieces
        else:
            log.error("Trying to use self.model which doesn't exist. Please, fit model before using.")
            exit(1)

    def __len__(self):
        return self.vocab_size

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load model from file
        """
        if self.load_path:
            log.info("Loading model {} from path: '{}'."
                     .format(self.__class__.__name__, self.load_path))
            model = spm.SentencePieceProcessor()
            try:
                model.Load(str(self.load_path))
            except Exception as e:
                log.error(e.__repr__())
                sys.exit(1)
            self.vocab_size = len(model)
        else:
            log.error('No load_path "{}" provided.'
                      .format(self.load_path))
            sys.exit(1)
        return model

    def fit(self, train_file_path: str, *args, **kwargs):
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

        self.load_path = os.path.join(self.save_path, 'bpe.model')
        self.model = self.load()
        self._set_encoding_type()

    @overrides
    def __call__(self, batch: Union[List[str], List[List[str]]], preprocess: bool = True, *args, **kwargs):
        """
        tokenize batch / or join list of tokens
        if batch is List[str] -> tokenize each str
        if batch is List[List[str]] -> join each List[str]

        """
        if not self.model:
            log.error("Trying to use self.model which doesn't exist. Please, fit model before using.")
            exit(1)

        if isinstance(batch[0], str):
            encoded = []
            if self.preprocess_call:
                for s in self._preprocess(batch):
                    encoded.append(self._encode(s))
            else:
                for s in batch:
                    encoded.append(self._encode(s))

            lengths = [len(sent) for sent in encoded]
            if self.padding:
                max_len = max(lengths)
                for i in range(len(encoded)):
                    encoded[i].extend(['<blank>']*(max_len-lengths[i]))
            return encoded, lengths

        elif isinstance(batch[0], list) and \
                ((isinstance(batch[0][0], str) and not self.indexes_only)
                 or (isinstance(batch[0][0], int) and self.indexes_only)):
            decoded = []
            for s in batch:
                decoded.append(self._decode(s))
            return decoded, [len(sent) for sent in decoded]

    def _preprocess(self, batch: List[str]):
        """
        turning strings into lowercase and remove punctuation
        :param batch: list of strings
        :yield: string
        """
        #TODO: remove symbols with hats
        for s in batch:
            yield s.lower().translate(self._preprocess_table)


