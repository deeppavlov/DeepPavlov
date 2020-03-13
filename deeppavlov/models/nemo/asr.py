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
from nemo_asr.helpers import post_process_predictions
from ruamel.yaml import YAML

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = logging.getLogger(__name__)


@register('nemo_asr')
class NeMoASR(Component, Serializable):
    def __init__(self,
                 load_path: Union[str, Path],
                 model_yaml: str,
                 encoder_checkpoint: str,
                 decoder_checkpoint: str,
                 **kwargs) -> None:
        super(NeMoASR, self).__init__(save_path=None, load_path=load_path, **kwargs)
        self._encoder_ckpt_path = self.load_path / encoder_checkpoint
        self._decoder_ckpt_path = self.load_path / decoder_checkpoint

        yaml = YAML(typ="safe")
        with open(self.load_path / model_yaml) as f:
            jasper_model_definition = yaml.load(f)

        self.labels = jasper_model_definition['labels']
        placement = nemo.core.DeviceType.GPU if torch.cuda.is_available() else nemo.core.DeviceType.CPU
        self.neural_factory = nemo.core.NeuralModuleFactory(placement=placement)
        self.data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            **jasper_model_definition['AudioPreprocessing'],
            sample_rate=jasper_model_definition['sample_rate']
        )
        self.jasper_encoder = nemo_asr.JasperEncoder(
            **jasper_model_definition['JasperEncoder'],
            feat_in=jasper_model_definition['AudioPreprocessing']['features'],
        )
        self.jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=jasper_model_definition['JasperEncoder']['jasper'][-1]['filters'],
            num_classes=len(self.labels)
        )
        self.greedy_decoder = nemo_asr.GreedyCTCDecoder()

        self.load()

    def save(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def load(self) -> None:
        self.jasper_encoder.restore_from(self._encoder_ckpt_path, local_rank=0)
        self.jasper_decoder.restore_from(self._decoder_ckpt_path, local_rank=0)

    def __call__(self, manifests):
        # TODO: don't forget to add batch_size parametrization (batchification are speeding up inference)
        manifest = ','.join(manifests)
        data_layer = nemo_asr.AudioToTextDataLayer(shuffle=False, manifest_filepath=manifest, labels=self.labels,
                                                   batch_size=1)

        audio_signal, audio_signal_len, _, _ = data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
        encoded, encoded_len = self.jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = self.jasper_decoder(encoder_output=encoded)
        predictions = self.greedy_decoder(log_probs=log_probs)
        eval_tensors = [predictions]
        tensors = self.neural_factory.infer(tensors=eval_tensors)
        prediction = post_process_predictions(tensors[0], self.labels)

        return prediction
