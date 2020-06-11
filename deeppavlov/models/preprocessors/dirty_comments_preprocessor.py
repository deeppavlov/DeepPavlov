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

import re
import string
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('dirty_comments_preprocessor')
class DirtyCommentsPreprocessor(Component):
    """
    Class implements preprocessing of english texts with low level of literacy such as comments
    """

    def __init__(self, remove_punctuation: bool = True, *args, **kwargs):
        self.remove_punctuation = remove_punctuation

    def __call__(self, batch: List[str], **kwargs) -> List[str]:
        """
        Preprocess given batch

        Args:
            batch: list of text samples
            **kwargs: additional arguments

        Returns:
            list of preprocessed text samples
        """
        f = [x.lower() for x in batch]
        f = [re.sub("<\S*>", " ", x) for x in f]
        f = [re.sub('\s+', ' ', x) for x in f]

        f = [x.replace("won't", "will not") for x in f]
        f = [x.replace("can't", "cannot") for x in f]
        f = [x.replace("i'm", "i am") for x in f]
        f = [x.replace(" im ", " i am ") for x in f]
        f = [x.replace("'re", " are") for x in f]
        f = [x.replace("ain't", "is not") for x in f]
        f = [x.replace("'ll", " will") for x in f]
        f = [x.replace("n't", " not") for x in f]
        f = [x.replace("'ve", " have") for x in f]
        f = [x.replace("'s", " is") for x in f]
        f = [x.replace("'d", " would") for x in f]

        f = [re.sub("ies( |$)", "y ", x) for x in f]
        f = [re.sub("s( |$)", " ", x) for x in f]
        f = [re.sub("ing( |$)", " ", x) for x in f]

        f = [x.replace(" u ", " you ") for x in f]
        f = [x.replace(" em ", " them ") for x in f]
        f = [x.replace(" da ", " the ") for x in f]
        f = [x.replace(" yo ", " you ") for x in f]
        f = [x.replace(" ur ", " your ") for x in f]
        f = [x.replace(" u r ", " you are ") for x in f]
        f = [x.replace(" urs ", " yours ") for x in f]
        f = [x.replace("y'all", "you all") for x in f]

        f = [x.replace(" r u ", " are you ") for x in f]
        f = [x.replace(" r you", " are you") for x in f]
        f = [x.replace(" are u ", " are you ") for x in f]

        f = [x.replace("\\n", " ") for x in f]
        f = [x.replace("\\t", " ") for x in f]
        f = [x.replace("\\xa0", " ") for x in f]
        f = [x.replace("\\xc2", " ") for x in f]
        f = [re.sub("[0-9]+", " 0 ", x) for x in f]

        f = [re.sub(r'([' + string.printable + r'])\1{3,}', r'\1\1', x).strip() for x in f]

        if self.remove_punctuation:
            f = [re.sub(r'([' + string.punctuation + '])', ' ', x) for x in f]

        f = [re.sub(' +', ' ', x) for x in f]
        return f
