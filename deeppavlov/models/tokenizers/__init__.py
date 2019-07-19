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
    from .lazy_tokenizer import LazyTokenizer
except ImportError:
    pass

try:
    from .nltk_moses_tokenizer import NLTKMosesTokenizer
except ImportError:
    pass

try:
    from .nltk_tokenizer import NLTKTokenizer
except ImportError:
    pass

try:
    from .ru_sent_tokenizer import RuSentTokenizer
except ImportError:
    pass

try:
    from .ru_tokenizer import RussianTokenizer
except ImportError:
    pass

try:
    from .spacy_tokenizer import StreamSpacyTokenizer
except ImportError:
    pass

try:
    from .split_tokenizer import SplitTokenizer
except ImportError:
    pass
