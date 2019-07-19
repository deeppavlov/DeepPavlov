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
    from .bert_classifier import BertClassifierModel
except ImportError:
    pass

try:
    from .bert_ner import BertNerModel
except ImportError:
    pass

try:
    from .bert_ranker import BertRankerModel, BertSepRankerModel, BertSepRankerPredictor
except ImportError:
    pass

try:
    from .bert_squad import BertSQuADInferModel, BertSQuADModel
except ImportError:
    pass
