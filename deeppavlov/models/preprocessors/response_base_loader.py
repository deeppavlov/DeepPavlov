# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable
from logging import getLogger
import numpy as np

logger = getLogger(__name__)


@register('response_base_loader')
class ResponseBaseLoader(Serializable):
    # TODO: docs

    def __init__(self, save_path, *args, **kwargs):
        super().__init__(save_path, *args, **kwargs)
        self.resps = None
        self.resp_vecs = None
        self.conts = None
        self.cont_vecs = None
        if self.load_path is not None:
            resp_file = self.load_path / "responses.csv"
            if resp_file.exists():
                with open(resp_file) as f:
                    responses = f.readlines()
                    self.resps = [el.strip('#\n') for el in responses]
            resp_vec_file = self.load_path / "resp_vecs.npy"
            if resp_vec_file.exists():
                self.resp_vecs = np.load(resp_vec_file)
            cont_file = self.load_path / "contexts.csv"
            if cont_file.exists():
                with open(cont_file) as f:
                    contexts = f.readlines()
                    self.conts = [el.strip('#\n') for el in contexts]
            cont_vec_file = self.load_path / "cont_vecs.npy"
            if cont_vec_file.exists():
                self.cont_vecs = np.load(cont_vec_file)

    def load(self):
        pass

    def save(self):
        pass