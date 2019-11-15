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


from logging import getLogger
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator

log = getLogger(__name__)


@register('dstc2_intents_iterator')
class Dstc2IntentsDatasetIterator(BasicClassificationDatasetIterator):
    """
    Class gets data dictionary from DSTC2DatasetReader instance, construct intents from act and slots, \
        merge fields if necessary, split a field if necessary

    Args:
        data: dictionary of data with fields "train", "valid" and "test" (or some of them)
        fields_to_merge: list of fields (out of ``"train", "valid", "test"``) to merge
        merged_field: name of field (out of ``"train", "valid", "test"``) to which save merged fields
        field_to_split: name of field (out of ``"train", "valid", "test"``) to split
        split_fields: list of fields (out of ``"train", "valid", "test"``) to which save splitted field
        split_proportions: list of corresponding proportions for splitting
        seed: random seed
        shuffle: whether to shuffle examples in batches
        *args: arguments
        **kwargs: arguments

    Attributes:
        data: dictionary of data with fields "train", "valid" and "test" (or some of them)
    """

    def __init__(self, data: dict,
                 fields_to_merge: List[str] = None, merged_field: str = None,
                 field_to_split: str = None, split_fields: List[str] = None, split_proportions: List[float] = None,
                 seed: int = None, shuffle: bool = True,
                 *args, **kwargs):
        """
        Initialize dataset using data from DatasetReader,
        merges and splits fields according to the given parameters
        """
        super().__init__(data, fields_to_merge, merged_field,
                         field_to_split, split_fields, split_proportions,
                         seed=seed, shuffle=shuffle)

        new_data = dict()
        new_data['train'] = []
        new_data['valid'] = []
        new_data['test'] = []

        for field in ['train', 'valid', 'test']:
            for turn in self.data[field]:
                reply = turn[0]
                curr_intents = []
                if reply['intents']:
                    for intent in reply['intents']:
                        for slot in intent['slots']:
                            if slot[0] == 'slot':
                                curr_intents.append(intent['act'] + '_' + slot[1])
                            else:
                                curr_intents.append(intent['act'] + '_' + slot[0])
                        if len(intent['slots']) == 0:
                            curr_intents.append(intent['act'])
                else:
                    if reply['text']:
                        curr_intents.append('unknown')
                    else:
                        continue
                new_data[field].append((reply['text'], curr_intents))

        self.data = new_data
