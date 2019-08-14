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
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done

log = getLogger(__name__)


@register('imdb_reader')
class ImdbReader(DatasetReader):
    """This class downloads and reads the IMDb sentiment classification dataset.

    https://ai.stanford.edu/~amaas/data/sentiment/

    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.
    (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association
    for Computational Linguistics (ACL 2011).
    """

    @overrides
    def read(self, data_path: str, url: Optional[str] = None,
             *args, **kwargs) -> Dict[str, List[Tuple[Any, Any]]]:
        """
        Args:
            data_path: A path to a folder with dataset files.
            url: A url to the archive with the dataset to download if the data folder is empty.
        """
        data_path = Path(data_path)

        if url is None:
            url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        if not is_done(data_path):
            log.info('[downloading data from {} to {}]'.format(url, data_path))
            download_decompress(url, data_path)
            mark_done(data_path)

        alternative_data_path = data_path / "aclImdb"
        if alternative_data_path.exists():
            data_path = alternative_data_path

        data = {"train": [],
                "test": []}
        for data_type in data.keys():
            for label in ["neg", "pos"]:
                labelpath = data_path / data_type / label
                if not labelpath.exists():
                    raise RuntimeError(f"Cannot load data: {labelpath} does not exist")
                for filename in labelpath.glob("*.txt"):
                    with filename.open(encoding='utf-8') as f:
                        text = f.read()
                    data[data_type].append((text, [label]))

            if not data[data_type]:
                raise RuntimeError(f"Could not load the '{data_type}' dataset, "
                                   "probably data dirs are empty")

        return data
