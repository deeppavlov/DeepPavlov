from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.data.dataset import Dataset
from deeppavlov.data.dataset_readers.dataset_reader import DatasetReader

from intent_recognition.intent_dataset import IntentDataset
from intent_recognition.intent_dataset_reader import IntentDatasetReader
from intent_recognition.utils import EmbeddingsDict
from intent_recognition.intent_models import KerasIntentModel

import os
import json


def main(config_name='config.json'):
    with open(config_name) as f:
        config = json.load(f)

    reader_config = config['dataset_reader']
    reader: DatasetReader = _REGISTRY[reader_config['name']]
    data = reader.read(train_data_path=reader_config.get('train_data_path'),
                       valid_data_path=reader_config.get('valid_data_path'),
                       test_data_path=reader_config.get('test_data_path'))

    dataset_config = config['dataset']
    dataset: Dataset = from_params(_REGISTRY[dataset_config['name']],
                                   dataset_config, data=data)

    intents = dataset.extract_classes()
    print("Considered intents:", intents)
    model_config = config['model']
    model: Trainable = from_params(_REGISTRY[model_config['name']],
                                   model_config, opt=model_config, classes=intents)

    print("Network parameters: ", model.network_params)
    print("Learning parameters:", model.learning_params)

    # if 'valid' in data.keys():
    #     print('___Validation set is given___')
    # elif 'val_split' in model.learning_params.keys():
    #     print('___Cross_validation split is given___')
    # else:
    #     print('___Validation set and cross validation split are not given.____\n____Cross validation split = 0.1____')
    #
    # updates = 0
    # steps = 0
    # val_loss = 1e100
    # epochs_done = 0
    #
    # while (batch_generator._epochs_done < model.learning_params['epochs']:
    #     batch, batch_labels = batch_generator.next_batch()
    #     model.train_on_batch(batch, batch_labels, class_weight=class_weights)
    #     if steps % 100 == 0:
    #         res = model.test_on_batch(batch, batch_labels)
    #         print("___Training batch___ %s: %f \t %s: %f \t %s: %f" %
    #               (metrics_names[0], res[0],
    #                metrics_names[1], res[1], metrics_names[2], res[2]))
    #
    #     if epochs_done < batch_generator._epochs_done:
    #         valid_res = model.test_on_batch(valid_batch, valid_batch_labels)
    #         print("___Validation___ %s: %f \t %s: %f \t %s: %f" %
    #               (metrics_names[0], valid_res[0],
    #                metrics_names[1], valid_res[1],
    #                metrics_names[2], valid_res[2]))
    #         if valid_res[0] > valid_loss_prev:
    #             validation_increase += 1
    #             print("__Validation loss increase. Impatience %d out of %d" % (
    #             validation_increase, VALIDATION_PATIENCE))
    #             if validation_increase == VALIDATION_PATIENCE:
    #                 print("__STOP: validation is out of patience___")
    #                 break
    #         valid_loss_prev = valid_res[0]
    #         epochs_done += 1
    #     steps += 1






    batch_gen = dataset.batch_generator(batch_size=64, data_type='train')
    for batch in batch_gen:
        model.train_on_batch(batch)

    model.save(fname='/home/dilyara/data/models/intent_models/dstc2/pilot_dstc2/intent_cnn_0')


if __name__ == '__main__':
    main()
