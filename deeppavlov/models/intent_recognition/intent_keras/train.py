from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.dataset_readers.intent_dataset_reader import IntentDatasetReader
from deeppavlov.datasets.intent_dataset import IntentDataset
from deeppavlov.models.intent_recognition.intent_keras.intent_model import KerasIntentModel
from deeppavlov.preprocessors.intent_preprocessor import IntentPreprocessor
from deeppavlov.models.intent_recognition.intent_keras.utils import labels2onehot, log_metrics, \
    proba2labels, proba2onehot

import sys
import json
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from intent_recognition.metrics import fmeasure
import keras.backend as K

def main(config_name='config.json'):

    # K.clear_session()

    with open(config_name) as f:
        config = json.load(f)

    # Reading datasets from files
    reader_config = config['dataset_reader']
    reader = _REGISTRY[reader_config['name']]
    data = reader.read(train_data_path=reader_config.get('train_data_path'),
                       valid_data_path=reader_config.get('valid_data_path'),
                       test_data_path=reader_config.get('test_data_path'))

    # Building dict of datasets
    dataset_config = config['dataset']
    dataset = from_params(_REGISTRY[dataset_config['name']],
                          dataset_config, data=data)

    # Merging train and valid dataset for further split on train/valid
    dataset.merge_data(fields_to_merge=['train', 'valid'], new_field='train')
    dataset.split_data(field_to_split='train', new_fields=['train', 'valid'], proportions=[0.9, 0.1])

    preproc_config = config['preprocessing']
    preproc = from_params(_REGISTRY[preproc_config['name']],
                                    preproc_config)
    dataset = preproc.preprocess(dataset=dataset, data_type='train')
    dataset = preproc.preprocess(dataset=dataset, data_type='valid')
    dataset = preproc.preprocess(dataset=dataset, data_type='test')

    # Extracting unique classes
    intents = dataset.extract_classes()
    print("Considered intents:", intents)

    # Initializing model
    model_config = config['model']
    model_config['classes'] = intents
    model = from_params(_REGISTRY[model_config['name']],
                        model_config)

    print("Considered:", model.metrics_names)

    if 'valid' in data.keys():
        print('___Validation set is given___')
    elif 'val_split' in model.opt.keys():
        print('___Validation split is given___')
    else:
        print('___Validation set and validation split are not given.____\n____Validation split = 0.1____')
        model.opt['val_split'] = 0.1
        dataset.split_data(field_to_split='train', new_fields=['train', 'valid'],
                           proportions=[1. - model.opt['val_split'],
                                        model.opt['val_split']])

    model.train(dataset)

    model.save(fname=model.opt['model_file'])

    test_batch_gen = dataset.batch_generator(batch_size=model.opt['batch_size'],
                                             data_type='test')
    test_preds = []
    test_true = []
    for test_id, test_batch in enumerate(test_batch_gen):
        test_preds.extend(model.infer(test_batch[0]))
        test_true.extend(labels2onehot(test_batch[1], model.classes))
        if model.opt['show_examples'] and test_id == 0:
            for j in range(model.opt['batch_size']):
                print(test_batch[0][j],
                      test_batch[1][j],
                      proba2labels([test_preds[j]], model.confident_threshold, model.classes))

    test_true = np.asarray(test_true, dtype='float64')
    test_preds = np.asarray(test_preds, dtype='float64')

    test_values = []
    test_values.append(log_loss(test_true, test_preds))
    test_values.append(accuracy_score(test_true, proba2onehot(test_preds, model.confident_threshold, model.classes)))
    test_values.append(fmeasure(test_true, proba2onehot(test_preds, model.confident_threshold, model.classes)))

    log_metrics(names=model.metrics_names,
                values=test_values,
                mode='test')


if __name__ == '__main__':
    main()
