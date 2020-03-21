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
from nemo.utils import get_checkpoint_from_dir
from nemo_asr.helpers import post_process_predictions
from ruamel.yaml import YAML

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = logging.getLogger(__name__)


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

    def __call__(self, manifest):
        data_layer = nemo_asr.AudioToTextDataLayer(manifest_filepath=manifest, **self.data_layer_kwargs)
        audio_paths = [d['audio_filepath'] for d in data_layer._dataset.manifest._data]

        audio_signal, audio_signal_len, _, _ = data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(input_signal=audio_signal,
                                                                        length=audio_signal_len)
        encoded, encoded_len = self.jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = self.jasper_decoder(encoder_output=encoded)
        predictions = self.greedy_decoder(log_probs=log_probs)
        eval_tensors = [predictions]
        tensors = self.neural_factory.infer(tensors=eval_tensors)
        prediction = post_process_predictions(tensors[0], self.labels)

        return audio_paths, prediction
