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

import base64
from datetime import datetime
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import Collection, Union

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('bytes_saver')
class BytesSaver(Component):
    """Saves Binary I/O objects to ROM."""
    def __init__(self, save_path: Union[str, Path], **kwargs) -> None:
        """ Initializes save path.

        Args:
            save_path: Path to the directory for saving files. If the directory does not exist, it will be created.

        """
        self.save_path = expand_path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, audio_batch: Collection[BytesIO], filename_batch: Collection[str] = None) -> Collection[str]:
        """Saves Binary I/O objects batch.

        Args:
            audio_batch: Binary I/O objects batch to save.
            filename_batch: Batch of file names. If batch is None timestamp and file index number are used as name with
                .wav extension.

        Returns:
            file_path_batch: Batch of paths to saved files.

        """
        audio_batch_len = len(audio_batch)
        if filename_batch is not None and audio_batch_len != len(filename_batch):
            error_msg = 'audio_batch length is not equal filename_batch length: '\
                        f'{audio_batch_len} != {len(filename_batch)}'
            log.error(error_msg)
            raise ValueError(error_msg)

        if filename_batch is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
            filename_batch = [f'{timestamp}_{i}.wav' for i in range(audio_batch_len)]

        file_path_batch = [str(self.save_path / file) for file in filename_batch]
        for audio, filename in zip(audio_batch, file_path_batch):
            with open(filename, 'wb') as fout:
                fout.write(audio.read())

        return file_path_batch


@register('ascii_to_bytesIO')
def ascii_to_bytes_io(batch: Union[str, Collection]):
    """Recursively searches for strings in the input batch and converts them into the base64-encoded bytes wrapped in
    Binary I/O objects.

    Args:
        batch: A string or an iterable container with strings at some level of nesting.

    Returns:
        The same structure where all strings are converted into the base64-encoded bytes wrapped in Binary I/O object.

    """
    if isinstance(batch, str):
        return BytesIO(base64.decodebytes(batch.encode()))

    return list(map(ascii_to_bytes_io, batch))


@register('bytesIO_to_ascii')
def bytes_io_to_ascii(batch: Union[BytesIO, Collection]):
    """Recursively searches for Binary I/O objects in the input batch and converts them into ASCII-strings.

    Args:
        batch: A BinaryIO object or an iterable container with BinaryIO objects at some level of nesting.

    Returns:
        The same structure where all BinaryIO objects are converted into strings.

    """
    if isinstance(batch, BytesIO):
        return base64.encodebytes(batch.read()).decode('ascii')

    return list(map(bytes_io_to_ascii, batch))
