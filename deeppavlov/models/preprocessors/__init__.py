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
    from .assemble_embeddings_matrix import EmbeddingsMatrixAssembler
except ImportError:
    pass

try:
    from .bert_preprocessor import BertPreprocessor, BertNerPreprocessor, \
        BertRankerPreprocessor, BertSepRankerPredictorPreprocessor, BertSepRankerPreprocessor
except ImportError:
    pass

try:
    from .capitalization import CapitalizationPreprocessor, LowercasePreprocessor
except ImportError:
    pass

try:
    from .char_splitter import CharSplitter
except ImportError:
    pass

try:
    from .dirty_comments_preprocessor import DirtyCommentsPreprocessor
except ImportError:
    pass

try:
    from .ecommerce_preprocess import EcommercePreprocess
except ImportError:
    pass

try:
    from .mask import Mask
except ImportError:
    pass

try:
    from .odqa_preprocessors import DocumentChunker, StringMultiplier
except ImportError:
    pass

try:
    from .random_embeddings_matrix import RandomEmbeddingsMatrix
except ImportError:
    pass

try:
    from .response_base_loader import ResponseBaseLoader
except ImportError:
    pass

try:
    from .russian_lemmatizer import PymorphyRussianLemmatizer
except ImportError:
    pass

try:
    from .sanitizer import Sanitizer
except ImportError:
    pass

try:
    from .siamese_preprocessor import SiamesePreprocessor
except ImportError:
    pass

try:
    from .squad_preprocessor import SquadAnsPostprocessor, SquadAnsPreprocessor, \
        SquadBertAnsPostprocessor, SquadBertAnsPreprocessor, SquadBertMappingPreprocessor, \
        SquadPreprocessor, SquadVocabEmbedder
except ImportError:
    pass

try:
    from .str_lower import StrLower
except ImportError:
    pass

try:
    from .str_token_reverser import StrTokenReverser, StrTokenReverserInfo
except ImportError:
    pass

try:
    from .str_utf8_encoder import StrUTF8Encoder, StrUTF8EncoderInfo
except ImportError:
    pass
