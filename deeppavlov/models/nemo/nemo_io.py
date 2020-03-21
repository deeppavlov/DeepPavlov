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

import json
import tempfile
from itertools import zip_longest
from logging import getLogger
from pathlib import Path

from nemo_asr.parts.manifest import ManifestBase
from scipy.io import wavfile

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


class TTSManifestGenerator(Component):
    def __init__(self, tempdir = None):
        self.tempdir = tempdir or Path(tempfile.gettempdir(), 'deeppavlov')
        self.tempdir.mkdir(parents=True, exist_ok=True)

    def __call__(self, text_batch, **kwargs):
        _, manifest_path = tempfile.mkstemp(suffix='.json', text=True, dir=self.tempdir)
        with open(manifest_path, 'w') as manifest:
            manifest.writelines([f"{json.dumps({'text': text})}\n" for text in text_batch])
        return manifest_path


class WAVSaver(Component):
    def __init__(self, wav_dir=None, manifest_path=None):
        # TODO: expand_path for custom paths
        self.tempdir = wav_dir or Path(tempfile.gettempdir(), 'deeppavlov')
        self.tempdir.mkdir(parents=True, exist_ok=True)
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
        self.manifest_path = manifest_path or self.tempdir

    def __call__(self, audio_batch, text_batch=None, **kwargs):
        manifest = []
        for audio, text in zip_longest(audio_batch, text_batch or [], fillvalue=''):
            manifest_line = {'audio_filepath': None, 'duration': 0.0, 'text': text}
            if audio is not None:
                source_rate, source_sig = wavfile.read(audio)
                duration_seconds = len(source_sig) / float(source_rate)
                _, wav_path = tempfile.mkstemp(suffix='.wav', text=False, dir=self.tempdir)
                with open(wav_path, 'wb') as file:
                    file.write(audio.read())
                manifest_line['audio_filepath'] = wav_path
                manifest_line['duration'] = duration_seconds
            manifest.append(manifest_line)
        _, manifest_path = tempfile.mkstemp(suffix='.json', text=True, dir=self.manifest_path)
        with open(manifest_path, 'w') as manifest_file:
            manifest_file.writelines([f'{json.dumps(line)}\n' for line in manifest])
        return manifest_path


class ManifestCleaner(Component):
    def __init__(self, remove_audio=True):
        self.remove_audio = remove_audio

    def __call__(self, manifest_path, **kwargs):
        manifest_path = expand_path(manifest_path)
        if self.remove_audio:
            for item in ManifestBase.json_item_gen([manifest_path]):
                filepath = item.get('audio_filepath')
                if filepath:
                    Path(filepath).unlink()
                else:
                    log.warning('Skipping audio file removing')
        manifest_path.unlink()
