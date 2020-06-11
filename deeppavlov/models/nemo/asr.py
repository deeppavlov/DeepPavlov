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
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import torch
from nemo.collections.asr import AudioToMelSpectrogramPreprocessor, JasperEncoder, JasperDecoderForCTC, GreedyCTCDecoder
from nemo.collections.asr.helpers import post_process_predictions
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.core.neural_types import AudioSignal, NeuralType, LengthsType
from nemo.utils.decorators import add_port_docs
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from deeppavlov.core.common.registry import register
from deeppavlov.models.nemo.common import CustomDataLayerBase, NeMoBase

log = logging.getLogger(__name__)


class AudioInferDataset(Dataset):
    def __init__(self, audio_batch: List[Union[str, BytesIO]], sample_rate: int, int_values: bool, trim=False) -> None:
        """Dataset reader for AudioInferDataLayer.

        Args:
            audio_batch: Batch to be read. Elements could be either paths to audio files or Binary I/O objects.
            sample_rate: Audio files sample rate.
            int_values: If true, load samples as 32-bit integers.
            trim: Trim leading and trailing silence from an audio signal if True.

        """
        self.audio_batch = audio_batch
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values)
        self.trim = trim

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Processes audio batch item and extracts features.

        Args:
            index: Audio batch item index.

        Returns:
            features: Audio file's extracted features tensor.
            features_length: Features length tensor.

        """
        sample = self.audio_batch[index]
        features = self.featurizer.process(sample, trim=self.trim)
        features_length = torch.tensor(features.shape[0]).long()

        return features, features_length

    def __len__(self) -> int:
        return len(self.audio_batch)


class AudioInferDataLayer(CustomDataLayerBase):
    """Data Layer for ASR pipeline inference."""

    @property
    @add_port_docs()
    def output_ports(self) -> Dict[str, NeuralType]:
        return {
            "audio_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "a_sig_length": NeuralType(tuple('B'), LengthsType())
        }

    def __init__(self, *,
                 audio_batch: List[Union[str, BytesIO]],
                 batch_size: int = 32,
                 sample_rate: int = 16000,
                 int_values: bool = False,
                 trim_silence: bool = False,
                 **kwargs) -> None:
        """Initializes Data Loader.

        Args:
            audio_batch: Batch to be read. Elements could be either paths to audio files or Binary I/O objects.
            batch_size: How many samples per batch to load.
            sample_rate: Target sampling rate for data. Audio files will be resampled to sample_rate if
                it is not already.
            int_values: If true, load data as 32-bit integers.
            trim_silence: Trim leading and trailing silence from an audio signal if True.

        """
        self._sample_rate = sample_rate

        dataset = AudioInferDataset(audio_batch=audio_batch, sample_rate=sample_rate, int_values=int_values,
                                    trim=trim_silence)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=self.seq_collate_fn)
        super(AudioInferDataLayer, self).__init__(dataset, dataloader, **kwargs)

    @staticmethod
    def seq_collate_fn(batch: Tuple[Tuple[Tensor], Tuple[Tensor]]) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Collates batch of audio signal and audio length, zero pads audio signal.

        Args:
            batch: A tuple of tuples of audio signals and signal lengths. This collate function assumes the signals
                are 1d torch tensors (i.e. mono audio).

        Returns:
            audio_signal: Zero padded audio signal tensor.
            audio_length: Audio signal length tensor.

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
class NeMoASR(NeMoBase):
    """ASR model on NeMo modules."""

    def __init__(self, load_path: Union[str, Path], nemo_params_path: Union[str, Path], **kwargs) -> None:
        """Initializes NeuralModules for ASR.

        Args:
            load_path: Path to a directory with pretrained checkpoints for JasperEncoder and JasperDecoderForCTC.
            nemo_params_path: Path to a file containig labels and params for AudioToMelSpectrogramPreprocessor,
                JasperEncoder, JasperDecoderForCTC and AudioInferDataLayer.

        """
        super(NeMoASR, self).__init__(load_path=load_path, nemo_params_path=nemo_params_path, **kwargs)

        self.labels = self.nemo_params['labels']

        self.data_preprocessor = AudioToMelSpectrogramPreprocessor(
            **self.nemo_params['AudioToMelSpectrogramPreprocessor']
        )
        self.jasper_encoder = JasperEncoder(**self.nemo_params['JasperEncoder'])
        self.jasper_decoder = JasperDecoderForCTC(num_classes=len(self.labels), **self.nemo_params['JasperDecoder'])
        self.greedy_decoder = GreedyCTCDecoder()
        self.modules_to_restore = [self.jasper_encoder, self.jasper_decoder]

        self.load()

    def __call__(self, audio_batch: List[Union[str, BytesIO]]) -> List[str]:
        """Transcripts audio batch to text.

        Args:
            audio_batch: Batch to be transcribed. Elements could be either paths to audio files or Binary I/O objects.

        Returns:
            text_batch: Batch of transcripts.

        """
        data_layer = AudioInferDataLayer(audio_batch=audio_batch, **self.nemo_params['AudioToTextDataLayer'])
        audio_signal, audio_signal_len = data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(input_signal=audio_signal,
                                                                        length=audio_signal_len)
        encoded, encoded_len = self.jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
        log_probs = self.jasper_decoder(encoder_output=encoded)
        predictions = self.greedy_decoder(log_probs=log_probs)
        eval_tensors = [predictions]
        tensors = self.neural_factory.infer(tensors=eval_tensors)
        text_batch = post_process_predictions(tensors[0], self.labels)

        return text_batch
