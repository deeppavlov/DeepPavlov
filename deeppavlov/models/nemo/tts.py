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

from logging import getLogger
from pathlib import Path
from typing import Union

import nemo
import nemo_asr
import nemo_tts
import torch
from io import BytesIO
from scipy.io import wavfile
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_yaml
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.nemo.vocoder import WaveGlow, GriffinLim

log = getLogger(__name__)

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
        num_labels = len(self.data_layer_kwargs['labels'])
        self.data_layer_kwargs.update(
            {
                'bos_id': num_labels,
                'eos_id': num_labels + 1,
                'pad_id': num_labels + 2
            }
        )

        if vocoder == "waveglow":
            self.vocoder = WaveGlow(**tacotron2_params["WaveGlowNM"])
            self.modules_to_restore.append(self.vocoder)

        elif vocoder == 'griffin-lim':
            self.vocoder = GriffinLim(**tacotron2_params['GriffinLim'])

        else:
            raise ValueError(f"'{vocoder} vocoder does not supported.'")

        self.load()

    def __call__(self, manifest_path):
        data_layer = nemo_asr.TranscriptDataLayer(
            path=manifest_path,
            **self.data_layer_kwargs
        )
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


if __name__ == '__main__':
    model = NeMoTTS('~/Downloads/tacotron2/tacotron2.yaml', '~/Downloads/tacotron2')
    audio = model('/data/nemo/workdir/gen.json')
    from deeppavlov.models.nemo.nemo_io import WAVSaver
    saver = WAVSaver()
    print(saver(audio))
