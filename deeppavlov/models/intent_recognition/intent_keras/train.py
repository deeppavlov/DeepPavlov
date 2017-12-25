from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.params import from_params
from deeppavlov.dataset_readers.intent_dataset_reader import IntentDatasetReader
from deeppavlov.datasets.intent_dataset import IntentDataset
from deeppavlov.models.intent_recognition.intent_keras.intent_model import KerasIntentModel
from deeppavlov.preprocessors.intent_preprocessor import IntentPreprocessor
from deeppavlov.models.intent_recognition.intent_keras.utils import labels2onehot, log_metrics, \
    proba2labels, proba2onehot


import json
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model


def main(config_name='intent_config.json'):


    with open(config_name) as f:
        config = json.load(f)

    # train_model_from_config(config_path=config_name)


    interact_model('intent_config_infer.json')


if __name__ == '__main__':
    main()
