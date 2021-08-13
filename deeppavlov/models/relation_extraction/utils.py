# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
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

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import register


@register('re_postprocessor')
class RePostprocessor:

    def __init__(self, rel2id_path: str, rel2label_path: str, **kwargs):
        self.rel2id_path = rel2id_path
        self.rel2label_path = rel2label_path
        self.rel2id = read_json(str(expand_path(self.rel2id_path)))
        self.id2rel = {rel_id: rel for rel, rel_id in self.rel2id.items()}
        self.rel2label = read_json(str(expand_path(self.rel2label_path)))
        
    def __call__(self, model_output):
        rel_labels_batch = []
        for predictions in model_output:
            rel_labels = []
            rel_indices = np.nonzero(predictions)[0]
            for index in rel_indices:
                if index == 0:
                    rel_labels.append("no relation")
                    continue
                rel_p = self.id2rel[index]
                if rel_p in self.rel2label:
                    rel_label = (rel_p, self.rel2label[rel_p])
                else:
                    rel_label = ("-", rel_p)
                rel_labels.append(rel_label)
            rel_labels_batch.append(rel_labels)
        return rel_labels_batch
