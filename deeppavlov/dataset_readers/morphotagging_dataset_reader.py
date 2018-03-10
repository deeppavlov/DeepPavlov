"""
Copyright 2018 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


WORD_COLUMN, POS_COLUMN, TAG_COLUMN = 1, 3, 5


def read_infile(infile, word_column=WORD_COLUMN, pos_column=POS_COLUMN,
                tag_column=TAG_COLUMN, max_sents=-1, read_only_words=False):
    answer, curr_word_sent, curr_tag_sent = [], [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if len(curr_word_sent) > 0:
                    if read_only_words:
                        curr_tag_sent = None
                    answer.append((curr_word_sent, curr_tag_sent))
                curr_tag_sent, curr_word_sent = [], []
                if len(answer) == max_sents:
                    break
                continue
            splitted = line.split("\t")
            index = splitted[0]
            if not index.isdigit():
                continue
            curr_word_sent.append(splitted[word_column])
            if not read_only_words:
                pos, tag = splitted[pos_column], splitted[tag_column]
                tag = pos if tag == "_" else "{},{}".format(pos, tag)
                curr_tag_sent.append(tag)
        if len(curr_word_sent) > 0:
            if read_only_words:
                curr_tag_sent = None
            answer.append((curr_word_sent, curr_tag_sent))
    return answer


@register('morphotagger_dataset_reader')
class MorphotaggerDatasetReader(DatasetReader):
    """
    Class to read training datasets in UD format
    """

    def read(self, data_path, language=None, data_types=None,
             is_filepath=False, **kwargs):
        """
        Reads UD dataset from data_path.

        data_path: str or list, can be either
            1. a directory containing files. The file for data_type 'mode'
            is then data_path / {language}-ud-{mode}.conllu
            2. a single file. Set is_filepath=True in this case.
            3. a list of files, containing the same number of items as data_types
        """
        if data_types is None:
            data_types = ["train", "dev"]
        elif isinstance(data_types, str):
            data_types = list(data_types)
        for data_type in data_types:
            if data_type not in ["train", "dev", "test"]:
                raise ValueError("Unknown data_type: {}, only train, dev and test "
                                 "datatypes are allowed".format(data_type))
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if isinstance(data_path, Path):
            if is_filepath:
                # path to a single file
                data_path = [data_path]
            else:
                # path to data directory
                if language is None:
                    raise ValueError("You must implicitly provide language "
                                     "when providing data directory as source")
                data_path = [data_path / "{}-ud-{}.conllu".format(language, mode)
                             for mode in data_types]
        else:
            data_path = [Path(data_path) for data_path in data_path]
        if len(data_path) != len(data_types):
            raise ValueError("The number of input files in data_path and data types "
                             "in data_types must be equal")
        for filepath in data_path:
            if not filepath.exists():
                raise ValueError("No file {} exists".format(filepath))
        data = {}
        for mode, filepath in zip(data_types, data_path):
            if mode == "dev":
                mode = "valid"
            data[mode] = read_infile(filepath, **kwargs)
        return data
