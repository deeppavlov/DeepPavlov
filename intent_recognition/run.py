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
from intent_models import KerasIntentModel

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

# opt = dict()
# opt['fasttext_model'] = os.path.join(path, 'dstc2_intent_model', 'dstc2_fasttext_model_100.bin')
# opt['embedding_dim'] = 100
# fasttext_model = EmbeddingsDict(opt, opt['embedding_dim'])
# print(fasttext_model.fasttext_model['foo'])


opt = {'kernel_sizes_cnn': "1 2 3",
       'filters_cnn' : 64,
       'embedding_size': 100,
       'lear_metrics': 'accuracy',
       'confident_threshold': 0.6,
       'model_from_saved': False,
       'optimizer': 'Adam',
       'lear_rate': 0.01,
       'lear_rate_decay': 0.1,
       'loss': 'binary_crossentropy',
       'fasttext_model': '/home/dilyara/data/data_files/dstc2/dstc2_intent_model/dstc2_fasttext_model_100.bin',
       'model_file': 'home/dilyara/data/models/intent_models/dstc2/pilot_model',
       'text_size': 15,
       'coef_reg_cnn': 1e-4,
       'coef_reg_den': 1e-4,
       'dropout_rate': 0.5,
       'dense_size': 50,
       'model_name': 'cnn_model',
       'batch_size': 64,
       }

train_batch_generator = dataset.batch_generator(batch_size=opt['batch_size'], data_type='train')
valid_batch_generator = dataset.batch_generator(batch_size=opt['batch_size'], data_type='valid')


model = KerasIntentModel(opt, intents)
updates = 0
for epoch in range(200):
    for batch in train_batch_generator:
        updates += 1
        model.train_on_batch(batch)

train_batch = (['price range', 'address', 'and post code', 'expensive', 'im looking for a moderate priced restaurant in the south part of town', 'cool whats the phone number', 'phone number', 'thank you good bye', 'north', 'what is the phone number', 'whats the address and phone number', 'steakhouse', 'north american food', 'is there anything else', 'is there anything else', 'thank you', 'thank you good bye', 'yes', 'phone number', 'id like thai food', 'italian', 'what about thai', 'whats the addre', 'i dont care', 'thank you good bye', 'serve hello food', 'south part of town', 'or', 'cheap restaurant in the south part of town', 'thank you good bye', 'breath thank you goodbye', 'steakhouse', 'and what type of food', 'swedish', 'what is the phone number', 'uh north side of town german food', 'catalan', 'price range', 'address', 'dont care', 'how about italian food', 'i dont know', 'phone number', 'what type of food', 'cheap restaurant', 'thank you good bye', 'noise', 'eirtrean', 'whats the address', 'phone number', 'okay and uh whats the address', 'okay thank you bye', 'in the center of town', 'thank you good bye', 'what is the address', 'thank you good bye', 'unintelligible', 'what is the phone number', 'um world food', 'thank you good bye', 'and the postal code', 'im looking for a cheap restaurant in the south part of town', 'french food', 'moderately'],
               [['request_pricerange'], ['request_addr'], ['request_postcode'], ['inform_pricerange'], ['inform_pricerange', 'inform_area'], ['request_phone'], ['request_phone'], ['thankyou', 'bye'], ['inform_area'], ['request_phone'], ['request_addr', 'request_phone'], ['inform_food'], ['inform_food'], ['reqalts'], ['reqalts'], ['thankyou'], ['thankyou', 'bye'], ['affirm'], ['request_phone'], ['inform_food'], ['inform_food'], ['reqalts', 'inform_food'], ['request_addr'], ['inform_this'], ['thankyou', 'bye'], ['hello'], ['inform_area'], ['unknown'], ['inform_pricerange', 'inform_area'], ['thankyou', 'bye'], ['thankyou', 'bye'], ['inform_food'], ['request_food'], ['inform_food'], ['request_phone'], ['inform_food', 'inform_area'], ['inform_food'], ['request_pricerange'], ['request_addr'], ['inform_this'], ['reqalts', 'inform_food'], ['inform_this'], ['request_phone'], ['request_food'], ['inform_pricerange'], ['thankyou', 'bye'], ['unknown'], ['inform_food'], ['request_addr'], ['request_phone'], ['request_addr'], ['thankyou', 'bye'], ['inform_area'], ['thankyou', 'bye'], ['request_addr'], ['thankyou', 'bye'], ['unknown'], ['request_phone'], ['inform_food'], ['thankyou', 'bye'], ['request_postcode'], ['inform_pricerange', 'inform_area'], ['inform_food'], ['inform_pricerange']])

preds = model.infer(train_batch[0])
pred_labels = model.proba2labels(preds)
for i in range(opt['batch_size']):
    print(train_batch[1][i], pred_labels[i])


