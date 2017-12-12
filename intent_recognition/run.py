"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

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
import os
print(os.getcwd())
from intent_dataset import IntentDataset
from intent_dataset_reader import IntentDatasetReader
import os
from utils import EmbeddingsDict

path = '/home/dilyara/data/data_files/dstc2'
data_reader = IntentDatasetReader()
data_dict = data_reader.read(train_data_path=os.path.join(path, 'dstc2_train.csv'),
                             valid_data_path=os.path.join(path, 'dstc2_valid.csv'),
                             test_data_path=os.path.join(path, 'dstc2_test.csv'))
print(data_dict['train'][0])
dataset = IntentDataset(data=data_dict)
intents = dataset.extract_classes()
print(intents, len(intents))

prep_train = dataset.preprocess(data_type='all')
print(prep_train[:100])

opt = dict()
opt['fasttext_model'] = os.path.join(path, 'dstc2_intent_model', 'dstc2_fasttext_model_100.bin')
opt['embedding_dim'] = 100
fasttext_model = EmbeddingsDict(opt, opt['embedding_dim'])
print(fasttext_model.fasttext_model['foo'])


