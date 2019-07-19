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
    from .cos_sim_classifier import CosineSimilarityClassifier
except ImportError:
    pass

try:
    from .keras_classification_model import KerasClassificationModel
except ImportError:
    pass

try:
    from .proba2labels import Proba2Labels
except ImportError:
    pass

try:
    from .ru_obscenity_classifier import RuObscenityClassifier
except ImportError:
    pass
