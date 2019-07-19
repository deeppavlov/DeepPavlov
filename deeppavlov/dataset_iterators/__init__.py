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
    from .basic_classification_iterator import BasicClassificationDatasetIterator
except ImportError:
    pass

try:
    from .dialog_iterator import DialogDatasetIterator, DialogDBResultDatasetIterator
except ImportError:
    pass

try:
    from .document_bert_ner_iterator import DocumentBertNerIterator
except ImportError:
    pass

try:
    from .dstc2_intents_iterator import Dstc2IntentsDatasetIterator
except ImportError:
    pass

try:
    from .dstc2_ner_iterator import Dstc2NerDatasetIterator
except ImportError:
    pass

try:
    from .elmo_file_paths_iterator import ELMoFilePathsIterator
except ImportError:
    pass

try:
    from .file_paths_iterator import FilePathsIterator
except ImportError:
    pass

try:
    from .kvret_dialog_iterator import KvretDialogDatasetIterator
except ImportError:
    pass

try:
    from .morphotagger_iterator import MorphoTaggerDatasetIterator
except ImportError:
    pass

try:
    from .ner_few_shot_iterator import NERFewShotIterator
except ImportError:
    pass

try:
    from .siamese_iterator import SiameseIterator
except ImportError:
    pass

try:
    from .snips_intents_iterator import SnipsIntentIterator
except ImportError:
    pass

try:
    from .snips_ner_iterator import SnipsNerIterator
except ImportError:
    pass

try:
    from .sqlite_iterator import SQLiteDataIterator
except ImportError:
    pass

try:
    from .squad_iterator import SquadIterator, MultiSquadIterator, MultiSquadRetrIterator
except ImportError:
    pass

try:
    from .typos_iterator import TyposDatasetIterator
except ImportError:
    pass
