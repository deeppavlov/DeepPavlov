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
    from. amazon_ecommerce_reader import AmazonEcommerceReader
except ImportError:
    pass

try:
    from .basic_classification_reader import BasicClassificationDatasetReader
except ImportError:
    pass

try:
    from .conll2003_reader import Conll2003DatasetReader
except ImportError:
    pass

try:
    from .dstc2_reader import DSTC2DatasetReader
except ImportError:
    pass

try:
    from .faq_reader import FaqDatasetReader
except ImportError:
    pass

try:
    from .file_paths_reader import FilePathsReader
except ImportError:
    pass

try:
    from .insurance_reader import InsuranceReader
except ImportError:
    pass

try:
    from .kbqa_reader import KBQAReader
except ImportError:
    pass

try:
    from .kvret_reader import KvretDatasetReader
except ImportError:
    pass

try:
    from .line_reader import LineReader
except ImportError:
    pass

try:
    from .morphotagging_dataset_reader import MorphotaggerDatasetReader
except ImportError:
    pass

try:
    from .odqa_reader import ODQADataReader
except ImportError:
    pass

try:
    from .ontonotes_reader import OntonotesReader
except ImportError:
    pass

try:
    from .paraphraser_pretrain_reader import ParaphraserPretrainReader
except ImportError:
    pass

try:
    from .paraphraser_reader import ParaphraserReader
except ImportError:
    pass

try:
    from .quora_question_pairs_reader import QuoraQuestionPairsReader
except ImportError:
    pass

try:
    from .siamese_reader import SiameseReader
except ImportError:
    pass

try:
    from .snips_reader import SnipsReader
except ImportError:
    pass

try:
    from .sq_reader import OntonotesReader
except ImportError:
    pass

try:
    from .squad_dataset_reader import SquadDatasetReader
except ImportError:
    pass

try:
    from .typos_reader import TyposCustom, TyposKartaslov, TyposWikipedia
except ImportError:
    pass

try:
    from .ubuntu_dstc7_mt_reader import UbuntuDSTC7MTReader
except ImportError:
    pass

try:
    from .ubuntu_v1_mt_reader import UbuntuV1MTReader
except ImportError:
    pass

try:
    from .ubuntu_v2_mt_reader import UbuntuV2MTReader
except ImportError:
    pass

try:
    from .ubuntu_v2_reader import UbuntuV2Reader
except ImportError:
    pass
