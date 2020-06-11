# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import torch
from nemo.collections.asr.parts import collections, parsers
from nemo.collections.asr.parts.dataset import TranscriptDataset
from nemo.collections.tts import TextEmbedding, Tacotron2Encoder, Tacotron2DecoderInfer, Tacotron2Postnet
from nemo.core.neural_types import NeuralType, LabelsType, LengthsType
from nemo.utils.decorators import add_port_docs
from nemo.utils.misc import pad_to
from scipy.io import wavfile
from torch import Tensor

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.models.nemo.common import CustomDataLayerBase, NeMoBase
from deeppavlov.models.nemo.vocoder import WaveGlow, GriffinLim

log = getLogger(__name__)


class TextDataset(TranscriptDataset):
    def __init__(self,
                 text_batch: List[str],
                 labels: List[str],
                 bos_id: Optional[int] = None,
                 eos_id: Optional[int] = None,
                 lowercase: bool = True) -> None:
        """Text dataset reader for TextDataLayer.

        Args:
            text_batch: Texts to be used for speech synthesis.
            labels: List of string labels to use when to str2int translation.
            bos_id: Label position of beginning of string symbol.
            eos_id: Label position of end of string symbol.
            lowercase: Whether to convert all uppercase characters in a text batch into lowercase characters.

        """
        parser = parsers.make_parser(labels, do_lowercase=lowercase)
        self.texts = collections.Text(text_batch, parser)
        self.bos_id = bos_id
        self.eos_id = eos_id


class TextDataLayer(CustomDataLayerBase):
    @property
    @add_port_docs()
    def output_ports(self) -> Dict[str, NeuralType]:
        return {
            'texts': NeuralType(('B', 'T'), LabelsType()),
            "texts_length": NeuralType(tuple('B'), LengthsType())
        }

    def __init__(self, *,
                 text_batch: List[str],
                 labels: List[str],
                 batch_size: int = 32,
                 bos_id: Optional[int] = None,
                 eos_id: Optional[int] = None,
                 pad_id: Optional[int] = None,
                 **kwargs) -> None:
        """A simple Neural Module for loading text data.

        Args:
            text_batch: Texts to be used for speech synthesis.
            labels: List of string labels to use when to str2int translation.
            batch_size: How many strings per batch to load.
            bos_id: Label position of beginning of string symbol. If None is initialized as `len(labels)`.
            eos_id: Label position of end of string symbol. If None is initialized as `len(labels) + 1`.
            pad_id: Label position of pad symbol. If None is initialized as `len(labels) + 2`.

        """
        len_labels = len(labels)
        if bos_id is None:
            bos_id = len_labels
        if eos_id is None:
            eos_id = len_labels + 1
        if pad_id is None:
            pad_id = len_labels + 2

        dataset = TextDataset(text_batch=text_batch, labels=labels, bos_id=bos_id, eos_id=eos_id)

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                 collate_fn=partial(self._collate_fn, pad_id=pad_id))
        super(TextDataLayer, self).__init__(dataset, dataloader, **kwargs)

    @staticmethod
    def _collate_fn(batch: Tuple[Tuple[Tensor], Tuple[Tensor]], pad_id: int) -> Tuple[Tensor, Tensor]:
        """Collates batch of texts.

        Args:
            batch: A tuple of tuples of audio signals and signal lengths.
            pad_id: Label position of pad symbol.

        Returns:
            texts: Padded texts tensor.
            texts_len: Text lengths tensor.

        """
        texts_list, texts_len = zip(*batch)
        max_len = max(texts_len)
        max_len = pad_to(max_len, 8)

        texts = torch.empty(len(texts_list), max_len, dtype=torch.long)
        texts.fill_(pad_id)

        for i, text in enumerate(texts_list):
            texts[i].narrow(0, 0, text.size(0)).copy_(text)

        if len(texts.shape) != 2:
            raise ValueError(f'Texts in collate function have shape {texts.shape}, should have 2 dimensions.')

        return texts, torch.stack(texts_len)


@register('nemo_tts')
class NeMoTTS(NeMoBase):
    """TTS model on NeMo modules."""
    def __init__(self,
                 load_path: Union[str, Path],
                 nemo_params_path: Union[str, Path],
                 vocoder: str = 'waveglow',
                 **kwargs) -> None:
        """Initializes NeuralModules for TTS.

        Args:
            load_path: Path to a directory with pretrained checkpoints for TextEmbedding, Tacotron2Encoder,
                Tacotron2DecoderInfer, Tacotron2Postnet and, if Waveglow vocoder is selected, WaveGlowInferNM.
            nemo_params_path: Path to a file containig sample_rate, labels and params for TextEmbedding,
                Tacotron2Encoder, Tacotron2Decoder, Tacotron2Postnet and TranscriptDataLayer.
            vocoder: Vocoder used to convert from spectrograms to audio. Available options: `waveglow` (needs pretrained
                checkpoint) and `griffin-lim`.

        """
        super(NeMoTTS, self).__init__(load_path=load_path, nemo_params_path=nemo_params_path, **kwargs)

        self.sample_rate = self.nemo_params['sample_rate']
        self.text_embedding = TextEmbedding(
            len(self.nemo_params['labels']) + 3,  # + 3 special chars
            **self.nemo_params['TextEmbedding']
        )
        self.t2_enc = Tacotron2Encoder(**self.nemo_params['Tacotron2Encoder'])
        self.t2_dec = Tacotron2DecoderInfer(**self.nemo_params['Tacotron2Decoder'])
        self.t2_postnet = Tacotron2Postnet(**self.nemo_params['Tacotron2Postnet'])
        self.modules_to_restore = [self.text_embedding, self.t2_enc, self.t2_dec, self.t2_postnet]

        if vocoder == 'waveglow':
            self.vocoder = WaveGlow(**self.nemo_params['WaveGlowNM'])
            self.modules_to_restore.append(self.vocoder)
        elif vocoder == 'griffin-lim':
            self.vocoder = GriffinLim(**self.nemo_params['GriffinLim'])
        else:
            raise ValueError(f'{vocoder} vocoder is not supported.')

        self.load()

    def __call__(self,
                 text_batch: List[str],
                 path_batch: Optional[List[str]] = None) -> Union[List[BytesIO], List[str]]:
        """Creates wav files or file objects with speech.

        Args:
            text_batch: Text from which human audible speech should be generated.
            path_batch: i-th element of `path_batch` is the path to save i-th generated speech file. If argument isn't
                specified, the synthesized speech will be stored to Binary I/O objects.

        Returns:
            List of Binary I/O objects with generated speech if `path_batch` was not specified, list of paths to files
                with synthesized speech otherwise.

        """
        if path_batch is None:
            path_batch = [BytesIO() for _ in text_batch]
        elif len(text_batch) != len(path_batch):
            raise ValueError('Text batch length differs from path batch length.')
        else:
            path_batch = [expand_path(path) for path in path_batch]

        data_layer = TextDataLayer(text_batch=text_batch, **self.nemo_params['TranscriptDataLayer'])
        transcript, transcript_len = data_layer()
        transcript_embedded = self.text_embedding(char_phone=transcript)
        transcript_encoded = self.t2_enc(char_phone_embeddings=transcript_embedded, embedding_length=transcript_len)
        mel_decoder, gate, alignments, mel_len = self.t2_dec(char_phone_encoded=transcript_encoded,
                                                             encoded_length=transcript_len)
        mel_postnet = self.t2_postnet(mel_input=mel_decoder)
        infer_tensors = [self.vocoder(mel_postnet), mel_len]
        evaluated_tensors = self.neural_factory.infer(tensors=infer_tensors)
        synthesized_batch = self.vocoder.get_audio(*evaluated_tensors)

        for fout, synthesized_audio in zip(path_batch, synthesized_batch):
            wavfile.write(fout, self.sample_rate, synthesized_audio)

        return path_batch
