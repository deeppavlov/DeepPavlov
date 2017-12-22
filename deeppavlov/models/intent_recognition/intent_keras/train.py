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
from metrics import fmeasure
from deeppavlov.core.commands.train import train_model_from_config

def main(config_name='intent_config.json'):


    with open(config_name) as f:
        config = json.load(f)

    train_model_from_config(config_path=config_name)


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
