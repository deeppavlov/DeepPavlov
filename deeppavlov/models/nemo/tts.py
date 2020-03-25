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
from typing import Union

import nemo
import nemo_tts
import torch
from nemo.backends.pytorch import DataLayerNM
from nemo.core import DeviceType
from nemo.core.neural_types import NeuralType, AxisType, BatchTag, TimeTag
from nemo.utils.misc import pad_to
from nemo_asr.parts.dataset import TranscriptDataset
from scipy.io import wavfile
from torch.utils.data import Dataset

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_yaml
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.nemo.vocoder import WaveGlow, GriffinLim

log = getLogger(__name__)


class TextDataset(TranscriptDataset):
    """A dataset class that reads and returns the text of a file.

    Args:
        path: (str) Path to file with newline separate strings of text
        labels (list): List of string labels to use when to str2int translation
        eos_id (int): Label position of end of string symbol
    """
    def __init__(self, texts, labels, bos_id=None, eos_id=None, lowercase=True):
        if lowercase:
            texts = [l.strip().lower() for l in texts]
        self.texts = texts

        self.char2num = {c: i for i, c in enumerate(labels)}
        self.bos_id = bos_id
        self.eos_id = eos_id


class TextDataLayer(DataLayerNM):
    """A simple Neural Module for loading textual transcript data.
    The path, labels, and eos_id arguments are dataset parameters.

    Args:
        pad_id (int): Label position of padding symbol
        batch_size (int): Size of batches to generate in data loader
        drop_last (bool): Whether we drop last (possibly) incomplete batch.
            Defaults to False.
        num_workers (int): Number of processes to work on data loading (0 for
            just main process).
            Defaults to 0.
    """

    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            'texts': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),

            "texts_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self,
                 texts,
                 labels,
                 batch_size=32,
                 bos_id=None,
                 eos_id=None,
                 pad_id=None,
                 drop_last=False,
                 num_workers=0,
                 shuffle=False,
                 **kwargs):
        super().__init__(**kwargs)

        len_labels = len(labels)
        if bos_id is None:
            bos_id = len_labels
        if eos_id is None:
            eos_id = len_labels + 1
        if pad_id is None:
            pad_id = len_labels + 2

        self._dataset = TextDataset(texts=texts, labels=labels, bos_id=bos_id, eos_id=eos_id)

        if self._placement == DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=partial(self._collate_fn, pad_id=pad_id, pad8=True),
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers
        )

    @staticmethod
    def _collate_fn(batch, pad_id, pad8=False):
        texts_list, texts_len = zip(*batch)
        max_len = max(texts_len)
        if pad8:
            max_len = pad_to(max_len, 8)

        texts = torch.empty(len(texts_list), max_len,
                            dtype=torch.long)
        texts.fill_(pad_id)

        for i, s in enumerate(texts_list):
            texts[i].narrow(0, 0, s.size(0)).copy_(s)

        if len(texts.shape) != 2:
            raise ValueError(
                f"Texts in collate function have shape {texts.shape},"
                f" should have 2 dimensions."
            )

        return texts, torch.stack(texts_len)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader


@register('nemo_tts')
class NeMoTTS(Component, Serializable):
    def __init__(self,
                 model_path: Union[str, Path],
                 checkpoints_dir: str,
                 vocoder: str = 'waveglow',
                 **kwargs):
        super(NeMoTTS, self).__init__(save_path=None, load_path=checkpoints_dir, **kwargs)
        placement = nemo.core.DeviceType.GPU if torch.cuda.is_available() else nemo.core.DeviceType.CPU
        self.neural_factory = nemo.core.NeuralModuleFactory(placement=placement)

        tacotron2_params = read_yaml(expand_path(model_path))
        self.sample_rate = tacotron2_params['sample_rate']

        self.text_embedding = nemo_tts.TextEmbedding(
            len(tacotron2_params["labels"]) + 3,  # + 3 special chars
            **tacotron2_params["TextEmbedding"])
        self.t2_enc = nemo_tts.Tacotron2Encoder(**tacotron2_params["Tacotron2Encoder"])
        self.t2_dec = nemo_tts.Tacotron2DecoderInfer(
            **tacotron2_params["Tacotron2Decoder"])
        self.t2_postnet = nemo_tts.Tacotron2Postnet(
            **tacotron2_params["Tacotron2Postnet"])
        self.modules_to_restore = [self.text_embedding, self.t2_enc, self.t2_dec, self.t2_postnet]
        self.data_layer_kwargs = tacotron2_params['TranscriptDataLayer']

        if vocoder == "waveglow":
            self.vocoder = WaveGlow(**tacotron2_params["WaveGlowNM"])
            self.modules_to_restore.append(self.vocoder)

        elif vocoder == 'griffin-lim':
            self.vocoder = GriffinLim(**tacotron2_params['GriffinLim'])

        else:
            raise ValueError(f"'{vocoder} vocoder does not supported.'")

        self.load()

    def __call__(self, texts):
        data_layer = TextDataLayer(texts, **self.data_layer_kwargs)
        transcript, transcript_len = data_layer()

        transcript_embedded = self.text_embedding(char_phone=transcript)
        transcript_encoded = self.t2_enc(
            char_phone_embeddings=transcript_embedded,
            embedding_length=transcript_len)
        mel_decoder, gate, alignments, mel_len = self.t2_dec(
            char_phone_encoded=transcript_encoded,
            encoded_length=transcript_len)
        mel_postnet = self.t2_postnet(mel_input=mel_decoder)
        infer_tensors = [self.vocoder(mel_postnet), mel_len]

        evaluated_tensors = self.neural_factory.infer(
            tensors=infer_tensors
        )

        data_batch = self.vocoder.get_audio(evaluated_tensors[0], evaluated_tensors[1])
        audio_batch = [BytesIO() for _ in data_batch]
        for audio, data in zip(audio_batch, data_batch):
            wavfile.write(audio, self.sample_rate, data)
        return audio_batch

    def load(self) -> None:
        checkpoints = nemo.utils.get_checkpoint_from_dir([str(module) for module in self.modules_to_restore], self.load_path)
        for module, checkpoint in zip(self.modules_to_restore, checkpoints):
            log.info(f'Restoring {module} from {checkpoint}')
            module.restore_from(checkpoint)

    def save(self, *args, **kwargs):
        pass
