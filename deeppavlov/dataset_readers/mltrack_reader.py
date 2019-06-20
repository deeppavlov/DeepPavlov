# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import re
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register("mltrack_reader")
class MLtrackReader(DatasetReader):
    """The class to read the MLtrack dataset from files.

    Please, see https://contest.yandex.ru/algorithm2018
    """

    def read(self,
             data_path: str,
             seed: int = None, *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the MLtrack dataset from files.

        Args:
            data_path: A path to a folder with dataset files.
            seed: Random seed.
        """

        data_path = expand_path(data_path)
        train_fname = data_path / "train.tsv"
        test_fname = data_path / "final.tsv"
        train_data = self.build_train_data(train_fname)
        valid_data = self.build_validation_data(train_fname, "valid")
        test_data = self.build_validation_data(test_fname, "test")
        dataset = {"train": train_data, "valid": valid_data, "test": test_data}
        return dataset

    def build_validation_data(self, name, data_type):
        lines = []
        with open(name, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                lines.append(line)
        if data_type == "test":
            for line in lines:
                line.extend(["bad", 1])
        pad = [""]
        num = 0
        y_voc = {"good": 2, "neutral": 1, "bad": 0}
        samples, labels, label = [], [], []
        cur_id = lines[0][0]
        context = self.clean_data(" ".join(lines[0][1:4]))
        sample = [context]
        for line in lines:
            context_id = line[0]
            context = self.clean_data(" ".join(line[1:4]))
            if context_id == cur_id:
                num += 1
                reply = self.clean_data(line[5])
                sample.append(reply)
                label.append(y_voc[line[-2]])
            else:
                sample += (6-num) * pad
                label += (6-num) * [0]
                samples.append(sample)
                labels.append(label)
                num = 1
                cur_id = line[0]
                reply = self.clean_data(line[5])
                sample = [context, reply]
                label = [y_voc[line[-2]]]
        sample += (6-num) * pad
        label += (6-num) * [0]
        samples.append(sample)
        labels.append(label)
        return list(zip(samples, labels))

    def build_train_data(self, name):
        lines = []
        with open(name, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                lines.append(line)
        samples, labels = [], []
        y_voc = {"good": 2, "neutral": 1, "bad": 0}
        for line in lines:
            context = self.clean_data(" ".join(line[1:4]))
            reply = self.clean_data(line[5])
            samples.append([context, reply])
            labels.append(y_voc[line[-2]])
        return list(zip(samples, labels))

    def clean_data(self, text):
        """Clean or fix invalid symbols in given text"""

        def clean(string):
            string = re.sub(
                '{ c : \$00ffff } |переведено специально для amovies . org сэм : йоу . '
                '|the long way home |♪ come here and be with me ♪ |( elevator bell ringing )'
                ' |_ bar _| синхронизация от elderman|sync , corrected by @ elder _ man '
                'переведено на cotranslate . net |== sync , corrected by elderman '
                '== @ elder _ man ',
                '', string)
            string = re.sub(
                '\*|~|†|•|¶|♪|#|\+|=|_|\. \. \. \. \.|\. \. \. \.|\. \. \.|\. \.',
                '', string)
            string = re.sub(r'\s+', ' ', string)
            string = re.sub('€', 'я', string)
            string = re.sub('¬', 'в', string)
            string = re.sub('\' \'', '"', string)
            string = re.sub('ьı', 'ы', string)
            string = re.sub('ƒ', 'д', string)
            string = re.sub('є', 'е', string)
            string = re.sub('ј', 'а', string)
            string = re.sub('ў', 'у', string)
            string = re.sub('∆', 'ж', string)
            string = re.sub('√', 'г', string)
            string = re.sub('≈', 'е', string)
            string = re.sub('- - - - - -', '–', string)
            string = re.sub('- - -', '–', string)
            string = re.sub('- -', '–', string)
            string = string.strip('`').strip()
            string = string.strip('–').strip()
            return string

        def m(string):
            string = re.sub('mне', 'мне', string)
            string = re.sub('mама', 'мама', string)
            string = re.sub('mистер', 'мистер', string)
            string = re.sub('mеня', 'меня', string)
            string = re.sub('mы', 'мы', string)
            string = re.sub('mэлоун', 'мэлоун', string)
            string = re.sub('mой', 'мой', string)
            string = re.sub('mисс', 'мисс', string)
            string = re.sub('mоя', 'моя', string)
            string = re.sub('mакфарленд', 'макфарленд', string)
            string = re.sub('m - м - м', 'м - м - м', string)
            string = re.sub('mм', 'мм', string)
            string = re.sub('mарек', 'марек', string)
            string = re.sub('maмa', 'мaмa', string)
            string = re.sub('mнe', 'мнe', string)
            string = re.sub('moй', 'мoй', string)
            string = re.sub('mapeк', 'мapeк', string)
            return string

        def s(string):
            string = re.sub('ѕодожди', 'подожди', string)
            string = re.sub('ѕока', 'пока', string)
            string = re.sub('ѕривет', 'привет', string)
            string = re.sub('ѕорис', 'борис', string)
            string = re.sub('ѕоже', 'боже', string)
            string = re.sub('ѕожалуйста', 'пожалуйста', string)
            string = re.sub('ѕилл', 'билл', string)
            string = re.sub('ѕерни', 'берни', string)
            string = re.sub('ѕольше', 'больше', string)
            string = re.sub('ѕочему', 'почему', string)
            string = re.sub('ѕрости', 'прости', string)
            string = re.sub('ѕольшое', 'большое', string)
            string = re.sub('ѕрайн', 'брайн', string)
            string = re.sub('ѕапа', 'папа', string)
            string = re.sub('ѕонимаешь', 'понимаешь', string)
            string = re.sub('ѕлагодарю', 'благодарю', string)
            string = re.sub('ѕогоди', 'погоди', string)
            string = re.sub('ѕо', 'по', string)
            string = re.sub('ѕойдем', 'пойдем', string)
            string = re.sub('ѕэм', 'пэм', string)
            string = re.sub('ѕотому', 'потому', string)
            string = re.sub('ѕонни', 'бонни', string)
            string = re.sub('ѕарень', 'парень', string)
            string = re.sub('ѕалермо', 'палермо', string)
            string = re.sub('ѕросай', 'бросай', string)
            string = re.sub('ѕеги', 'беги', string)
            string = re.sub('ѕошли', 'пошли', string)
            string = re.sub('ѕослушайте', 'послушайте', string)
            string = re.sub('ѕриятно', 'приятно', string)
            return string

        def n(string):
            string = re.sub('ќтойди', 'отойди', string)
            string = re.sub('ќна', 'она', string)
            string = re.sub('ќ !', 'о !', string)
            string = re.sub('ќ ,', 'о ,', string)
            string = re.sub('ќпять', 'опять', string)
            string = re.sub('ќк', 'ок', string)
            string = re.sub('ќни', 'они', string)
            string = re.sub('ќуе', 'оуе', string)
            string = re.sub('ќн', 'он', string)
            string = re.sub('ќ', 'н', string)
            return string

        def jo(string):
            string = re.sub('ёто', 'это', string)
            string = re.sub('ёдди', 'эдди', string)
            string = re.sub('ённи', 'энни', string)
            string = re.sub('ёйприл', 'эйприл', string)
            return string

        def cav(string):
            string = re.sub('" ак', 'так', string)
            string = re.sub('" звини', 'извини', string)
            string = re.sub('" ерт !', 'берт !', string)
            string = re.sub('" еперь', 'теперь', string)
            string = re.sub('" о', 'то', string)
            string = re.sub('сегодня "', 'сегодня ?', string)
            string = re.sub('днем "', 'днем ', string)
            string = re.sub('" олько', 'только', string)
            string = re.sub('" ы', 'ты', string)
            string = re.sub('" йди', 'уйди', string)
            string = re.sub('ц не', 'не', string)
            string = re.sub('ц ѕонимаешь', 'понимаешь', string)
            string = re.sub('ц  пока', 'пока', string)
            string = re.sub('" аппи', 'паппи', string)
            string = re.sub('ц " ,', '', string)
            return string

        def h(string):
            string = re.sub('ћадно', 'ладно', string)
            string = re.sub('ћюди', 'люди', string)
            string = re.sub('ћ', 'м', string)
            return string

        def ch(string):
            string = re.sub('ч да', 'да', string)
            string = re.sub('ч ћожет', 'может', string)
            string = re.sub('ч ну', 'ну', string)
            string = re.sub('ч ага', 'ага', string)
            return string

        clean_text = ch(h(cav(jo(n(s(m(clean(text))))))))

        return clean_text




