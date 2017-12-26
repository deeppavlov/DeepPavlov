import json
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
import keras.backend as K

from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.dataset_readers.intent_dataset_reader import IntentDatasetReader
from deeppavlov.datasets.intent_dataset import IntentDataset
from deeppavlov.models.intent_recognition.intent_keras.intent_model import KerasIntentModel
from deeppavlov.preprocessors.intent_preprocessor import IntentPreprocessor
from deeppavlov.models.intent_recognition.intent_keras.utils import labels2onehot, proba2onehot, \
    proba2labels, log_metrics
from deeppavlov.models.intent_recognition.intent_keras.metrics import fmeasure


def main(config_name='intent_config_infer.json'):

    # K.clear_session()

    with open(config_name) as f:
        config = json.load(f)

    # Reading datasets from files
    reader_config = config['dataset_reader']
    reader = _REGISTRY[reader_config['name']]
    data = reader.read(reader_config['data_path'])

    # Building dict of datasets
    dataset_config = config['dataset']
    dataset = from_params(_REGISTRY[dataset_config['name']],
                          dataset_config, data=data)

    # Merging train and valid dataset for further split on train/valid
    # dataset.merge_data(fields_to_merge=['train', 'valid'], new_field='train')
    # dataset.split_data(field_to_split='train', new_fields=['train', 'valid'], proportions=[0.9, 0.1])

    preproc_config = config['preprocessing']
    preproc = from_params(_REGISTRY[preproc_config['name']],
                                    preproc_config)
    # dataset = preproc.preprocess(dataset=dataset, data_type='train')
    # dataset = preproc.preprocess(dataset=dataset, data_type='valid')
    dataset = preproc.preprocess(dataset=dataset, data_type='test')

    # Extracting unique classes


    # Initializing model
    model_config = config['model']
    model = from_params(_REGISTRY[model_config['name']],
                        model_config)

    print("Considered loss and metrics: {}".format(model.metrics_names))

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
