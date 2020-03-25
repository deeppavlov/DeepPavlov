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

import logging
from pathlib import Path
from typing import Union

import nemo
import nemo_asr
import torch
from nemo.backends.pytorch import DataLayerNM
from nemo.core.neural_types import NeuralType, AxisType, BatchTag, TimeTag
from nemo.utils import get_checkpoint_from_dir
from nemo_asr.helpers import post_process_predictions
from nemo_asr.parts.features import WaveformFeaturizer
from ruamel.yaml import YAML
from torch.utils.data import Dataset

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = logging.getLogger(__name__)


class AudioInferDataset(Dataset):
    def __init__(self, audio,
            featurizer,
            trim=False):
        self.audio = audio
        self.featurizer = featurizer
        self.trim = trim

    def __getitem__(self, index):
        sample = self.audio[index]
        features = self.featurizer.process(sample,
                                           trim=self.trim)
        features_length = torch.tensor(features.shape[0]).long()

        return features, features_length

    def __len__(self):
        return len(self.audio)


class AudioDataLayer(DataLayerNM):
    @staticmethod
    def create_ports():
        input_ports = {}
        output_ports = {
            "audio_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(TimeTag)}),

            "a_sig_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            audio,
            batch_size=32,
            sample_rate=16000,
            int_values=False,
            trim_silence=False,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            **kwargs
    ):
        super().__init__(**kwargs)

        featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values)

        # Set up dataset
        self._dataset = AudioInferDataset(audio, featurizer, trim_silence)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self.seq_collate_fn,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader

    @staticmethod
    def seq_collate_fn(batch):
        """collate batch of audio sig, audio len, tokens, tokens len

        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                   LongTensor):  A tuple of tuples of signal, signal lengths,
                   encoded tokens, and encoded tokens length.  This collate func
                   assumes the signals are 1d torch tensors (i.e. mono audio).

        """
        _, audio_lengths = zip(*batch)
        max_audio_len = 0
        has_audio = audio_lengths[0] is not None
        if has_audio:
            max_audio_len = max(audio_lengths).item()

        audio_signal = []
        for sig, sig_len in batch:
            if has_audio:
                sig_len = sig_len.item()
                if sig_len < max_audio_len:
                    pad = (0, max_audio_len - sig_len)
                    sig = torch.nn.functional.pad(sig, pad)
                audio_signal.append(sig)

        if has_audio:
            audio_signal = torch.stack(audio_signal)
            audio_lengths = torch.stack(audio_lengths)
        else:
            audio_signal, audio_lengths = None, None

        return audio_signal, audio_lengths


@register('nemo_asr')
class NeMoASR(Component, Serializable):
    def __init__(self,
                 model_path: Union[str, Path],
                 checkpoints_dir: str,
                 **kwargs) -> None:
        super(NeMoASR, self).__init__(save_path=None, load_path=checkpoints_dir, **kwargs)

        yaml = YAML(typ="safe")
        with open(expand_path(model_path)) as f:
            jasper_model_definition = yaml.load(f)

        self.labels = jasper_model_definition['labels']
        placement = nemo.core.DeviceType.GPU if torch.cuda.is_available() else nemo.core.DeviceType.CPU
        self.neural_factory = nemo.core.NeuralModuleFactory(placement=placement)
        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            **jasper_model_definition['AudioToMelSpectrogramPreprocessor']
        )
        self.jasper_encoder = nemo_asr.JasperEncoder(
            **jasper_model_definition['JasperEncoder']
        )
        self.jasper_decoder = nemo_asr.JasperDecoderForCTC(
            num_classes=len(self.labels),
            **jasper_model_definition['JasperDecoder']
        )
        self.greedy_decoder = nemo_asr.GreedyCTCDecoder()
        self.modules_to_restore = [self.jasper_encoder, self.jasper_decoder]
        self.data_layer_kwargs = jasper_model_definition['AudioToTextDataLayer']

        self.load()

    def save(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def load(self) -> None:
        checkpoints = get_checkpoint_from_dir([str(module) for module in self.modules_to_restore], self.load_path)
        for module, checkpoint in zip(self.modules_to_restore, checkpoints):
            log.info(f'Restoring {module} from {checkpoint}')
            module.restore_from(checkpoint)

    def __call__(self, audio):
        data_layer = AudioDataLayer(audio=audio, **self.data_layer_kwargs)

        audio_signal, audio_signal_len = data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(input_signal=audio_signal,
                                                                        length=audio_signal_len)
        encoded, encoded_len = self.jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = self.jasper_decoder(encoder_output=encoded)
        predictions = self.greedy_decoder(log_probs=log_probs)
        eval_tensors = [predictions]
        tensors = self.neural_factory.infer(tensors=eval_tensors)
        prediction = post_process_predictions(tensors[0], self.labels)

        return prediction
