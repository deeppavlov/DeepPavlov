# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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


from deeppavlov.core.common.registry import register
import copy, re

@register('intent_preprocessing')
class IntentPreprocessor(object):
    """
    Class preprocesses data
    """
    @staticmethod
    def preprocess(dataset=None, data_type='train', data=None, *args, **kwargs):
        """
        Method preprocesses given data or field of the given dataset
        Args:
            dataset: dataset which field will be preprocessed
            data_type: field name that will be preprocessed
            data: optional, list of texts that will be preprocessed
            *args:
            **kwargs:

        Returns:
            preprocessed data
        """
        if data is not None:
            prep_data = IntentPreprocessor._preprocessing(data)
            return prep_data
        else:
            dataset_copy = copy.deepcopy(dataset)
            all_data = dataset.iter_all(data_type=data_type)
            texts = []
            for i, sample in enumerate(all_data):
                dataset_copy.data[data_type][i] = (IntentPreprocessor._preprocessing([sample[0]])[0],
                                                   dataset.data[data_type][i][1])
            return dataset_copy

    @staticmethod
    def _preprocessing(data):
        """
        Method contains preprocessing of the texts in the given data
        Args:
            data: list of texts

        Returns:
            list of the preprocessed texts
        """
        f = [x.lower() for x in data]
        f = [x.replace("\\n", " ") for x in f]
        f = [x.replace("\\t", " ") for x in f]
        f = [x.replace("\\xa0", " ") for x in f]
        f = [x.replace("\\xc2", " ") for x in f]

        f = [re.sub('!!+', ' !! ', x) for x in f]
        f = [re.sub('!', ' ! ', x) for x in f]
        f = [re.sub('! !', '!!', x) for x in f]

        f = [re.sub('\?\?+', ' ?? ', x) for x in f]
        f = [re.sub('\?', ' ? ', x) for x in f]
        f = [re.sub('\? \?', '??', x) for x in f]

        f = [re.sub('\?!+', ' ?! ', x) for x in f]

        f = [re.sub('\.\.+', '..', x) for x in f]
        f = [re.sub('\.', ' . ', x) for x in f]
        f = [re.sub('\.  \.', '..', x) for x in f]

        f = [re.sub(',', ' , ', x) for x in f]
        f = [re.sub(':', ' : ', x) for x in f]
        f = [re.sub(';', ' ; ', x) for x in f]
        f = [re.sub('\%', ' % ', x) for x in f]

        f = [x.replace("won't", "will not") for x in f]
        f = [x.replace("can't", "cannot") for x in f]
        f = [x.replace("i'm", "i am") for x in f]
        f = [x.replace(" im ", " i am ") for x in f]
        f = [x.replace("you're ", "you are") for x in f]
        f = [x.replace("'re", " are") for x in f]
        f = [x.replace("ain't", "is not") for x in f]
        f = [x.replace("'ll", " will") for x in f]
        f = [x.replace("'t", " not") for x in f]
        f = [x.replace("'ve", " have") for x in f]
        f = [x.replace("'s", " is") for x in f]
        f = [x.replace("'re", " are") for x in f]
        f = [x.replace("'d", " would") for x in f]

        f = [re.sub("ies( |$)", "y ", x) for x in f]
        f = [re.sub("s( |$)", " ", x) for x in f]
        f = [re.sub("ing( |$)", " ", x) for x in f]
        f = [x.replace("tard ", " ") for x in f]

        f = [re.sub(" [*$%&#@][*$%&#@]+", " xexp ", x) for x in f]
        f = [re.sub(" [0-9]+ ", " DD ", x) for x in f]
        f = [re.sub("<\S*>", "", x) for x in f]
        f = [re.sub('\s+', ' ', x) for x in f]
        return f
