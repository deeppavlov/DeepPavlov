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

from datetime import datetime
from logging import getLogger

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('wav_saver')
class WAVSaver(Component):
    def __init__(self, wav_dir, **kwargs):
        self.wav_dir = expand_path(wav_dir)
        self.wav_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, audio_batch, text_batch=None, **kwargs):
        audio_batch_len = len(audio_batch)
        if text_batch is not None and audio_batch_len != len(text_batch):
            raise ValueError('audio batch len != text batch len!')
        if text_batch is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
            text_batch = [f'{timestamp}_{i}.wav' for i in range(audio_batch_len)]
        for audio, name in zip(audio_batch, text_batch):
            with open(str(self.wav_dir / name), 'wb') as fout:
                fout.write(audio.read())
        return text_batch