from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.data.dataset import Dataset
from deeppavlov.data.dataset_readers.dataset_reader import DatasetReader

from intent_recognition.intent_dataset import IntentDataset
from intent_recognition.intent_dataset_reader import IntentDatasetReader
from intent_recognition.utils import EmbeddingsDict
from intent_recognition.intent_models import KerasIntentModel

import metrics as metrics_file
import keras.metrics as keras_metrics_file
import keras.backend as K
import os, sys
import json
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from metrics import fmeasure

def log_metrics(names, values, updates=None, mode='train'):
    sys.stdout.write("\r")  # back to previous line
    print("%s -->\t" % mode, end="")
    if updates is not None:
        print("updates: %d\t" % updates, end="")

    for id in range(len(names)):
        print("%s: %f\t" % (names[id], values[id]), end="")
    print(" ")

def main(config_name='config.json'):
    with open(config_name) as f:
        config = json.load(f)

    reader_config = config['dataset_reader']
    reader = _REGISTRY[reader_config['name']]
    data = reader.read(train_data_path=reader_config.get('train_data_path'),
                       valid_data_path=reader_config.get('valid_data_path'),
                       test_data_path=reader_config.get('test_data_path'))

    dataset_config = config['dataset']
    dataset = from_params(_REGISTRY[dataset_config['name']],
                                   dataset_config, data=data)

    intents = dataset.extract_classes()
    print("Considered intents:", intents)
    model_config = config['model']
    model = from_params(_REGISTRY[model_config['name']],
                                   model_config, opt=model_config, classes=intents)

    print("Network parameters: ", model.network_params)
    print("Learning parameters:", model.learning_params)

    if 'valid' in data.keys():
        print('___Validation set is given___')
    elif 'val_split' in model.learning_params.keys():
        print('___Cross_validation split is given___')
    else:
        print('___Validation set and cross validation split are not given.____\n____Cross validation split = 0.1____')
        model.learning_params['val_split'] = 0.1

    updates = 0
    val_loss = 1e100
    val_increase = 0
    epochs_done = 0

    print("Considered:", model.metrics_names)
    print('\n____Training____\n\n')

    while epochs_done < model.learning_params['epochs']:
        batch_gen = dataset.batch_generator(batch_size=model.learning_params['batch_size'],
                                            data_type='train')
        for step, batch in enumerate(batch_gen):
            model.train_on_batch(batch)
            updates += 1

            # log on training batch
            if model.learning_params['verbose'] and step % 500 == 0:
                batch_preds = model.infer(batch[0])
                # TODO: как сделать вывод в одну строчку во время данной эпохи?
                log_metrics(names=model.metrics_names,
                            values=model.metrics_values, updates=updates, mode='train')

        epochs_done += 1
        if epochs_done % model.learning_params['val_every_n_epochs'] == 0:
            if 'valid' in data.keys():
                valid_batch_gen = dataset.batch_generator(batch_size=model.learning_params['batch_size'],
                                                          data_type='valid')
                valid_preds = []
                valid_true = []
                for valid_id, valid_batch in enumerate(valid_batch_gen):
                    valid_preds.extend(model.infer(valid_batch[0]))
                    valid_true.extend(model.labels2onehot(valid_batch[1]))
                    if model_config['show_examples'] and valid_id == 0:
                        for j in range(model.learning_params['batch_size']):
                            print(valid_batch[0][j],
                                  valid_batch[1][j],
                                  model.proba2labels([valid_preds[j]]))

                valid_true = np.asarray(valid_true, dtype='float64')
                valid_preds = np.asarray(valid_preds, dtype='float64')

                valid_values = []
                valid_values.append(log_loss(valid_true, valid_preds))
                valid_values.append(accuracy_score(valid_true, model.proba2onehot(valid_preds)))
                valid_values.append(fmeasure(valid_true, model.proba2onehot(valid_preds)))

                log_metrics(names=model.metrics_names,
                            values=valid_values,
                            mode='valid')
                if valid_values[0] > val_loss:
                    val_increase += 1
                    print("__Validation impatience %d out of %d" % (
                        val_increase, model.learning_params['val_patience']))
                    if val_increase == model.learning_params['val_patience']:
                        print("___Stop training: validation is out of patience___")
                        break
                val_loss = valid_values[0]


        print('epochs_done: %d' % epochs_done)






    batch_gen = dataset.batch_generator(batch_size=64, data_type='train')
    for batch in batch_gen:
        model.train_on_batch(batch)

    model.save(fname='/home/dilyara/data/models/intent_models/dstc2/pilot_dstc2/intent_cnn_0')


if __name__ == '__main__':
    main()
