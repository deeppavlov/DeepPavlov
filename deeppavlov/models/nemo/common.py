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
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import Union

import nemo
import torch
from nemo.backends.pytorch import DataLayerNM
from torch.utils.data import Dataset, DataLoader

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_yaml
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


@register('base64_decode_bytesIO')
def ascii_to_bytes_io(batch: Union[str, list]) -> Union[BytesIO, list]:
    """Recursively searches for strings in the input batch and converts them into the base64-encoded bytes wrapped in
    Binary I/O objects.

    Args:
        batch: A string or an iterable container with strings at some level of nesting.

    Returns:
        The same structure where all strings are converted into the base64-encoded bytes wrapped in Binary I/O objects.

    """
    if isinstance(batch, str):
        return BytesIO(base64.decodebytes(batch.encode()))

    return list(map(ascii_to_bytes_io, batch))


@register('bytesIO_encode_base64')
def bytes_io_to_ascii(batch: Union[BytesIO, list]) -> Union[str, list]:
    """Recursively searches for Binary I/O objects in the input batch and converts them into ASCII-strings.

    Args:
        batch: A BinaryIO object or an iterable container with BinaryIO objects at some level of nesting.

    Returns:
        The same structure where all BinaryIO objects are converted into strings.

    """
    if isinstance(batch, BytesIO):
        return base64.encodebytes(batch.read()).decode('ascii')

    return list(map(bytes_io_to_ascii, batch))


class NeMoBase(Component, Serializable):
    """Base class for NeMo Chainer's pipeline components."""

    def __init__(self, load_path: Union[str, Path], nemo_params_path: Union[str, Path], **kwargs) -> None:
        """Initializes NeuralModuleFactory on CPU or GPU and reads nemo modules params from yaml.

        Args:
            load_path: Path to a directory with pretrained checkpoints for NeMo modules.
            nemo_params_path: Path to a file containig NeMo modules params.

        """
        super(NeMoBase, self).__init__(save_path=None, load_path=load_path, **kwargs)
        placement = nemo.core.DeviceType.GPU if torch.cuda.is_available() else nemo.core.DeviceType.CPU
        self.neural_factory = nemo.core.NeuralModuleFactory(placement=placement)
        self.modules_to_restore = []
        self.nemo_params = read_yaml(expand_path(nemo_params_path))

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def load(self) -> None:
        """Loads pretrained checkpoints for modules from self.modules_to_restore list."""
        module_names = [str(module) for module in self.modules_to_restore]
        checkpoints = nemo.utils.get_checkpoint_from_dir(module_names, self.load_path)
        for module, checkpoint in zip(self.modules_to_restore, checkpoints):
            log.info(f'Restoring {module} from {checkpoint}')
            module.restore_from(checkpoint)

    def save(self, *args, **kwargs) -> None:
        pass


class CustomDataLayerBase(DataLayerNM):
    def __init__(self, dataset: Dataset, dataloader: DataLoader, **kwargs) -> None:
        super(CustomDataLayerBase, self).__init__()
        self._dataset = dataset
        self._dataloader = dataloader

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> None:
        return None

    @property
    def data_iterator(self) -> torch.utils.data.DataLoader:
        return self._dataloader
