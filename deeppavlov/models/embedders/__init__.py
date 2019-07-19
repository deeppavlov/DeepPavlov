# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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


try:
    from .abstract_embedder import Embedder
except ImportError:
    pass

try:
    from .bow_embedder import BoWEmbedder
except ImportError:
    pass

try:
    from .elmo_embedder import ELMoEmbedder
except ImportError:
    pass

try:
    from .fasttext_embedder import FasttextEmbedder
except ImportError:
    pass

try:
    from .glove_embedder import GloVeEmbedder
except ImportError:
    pass

try:
    from .tfidf_weighted_embedder import TfidfWeightedEmbedder
except ImportError:
    pass
