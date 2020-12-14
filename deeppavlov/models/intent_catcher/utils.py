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

import numpy as np

def batch_samples(sentences, labels, batch_size=64):
    """Simple batcher"""
    assert isinstance(sentences, list) or isinstance(sentences, np.array), \
        print("`sentences` type must be list or np.array")
    assert isinstance(labels, list) or isinstance(labels, np.array), \
        print("`sentences` type must be list or np.array")
    i = 0
    while i < len(sentences):
        yield sentences[i:i+batch_size], labels[i:i+batch_size]
        i+=batch_size
