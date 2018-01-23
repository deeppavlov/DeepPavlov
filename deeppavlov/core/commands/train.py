import datetime
import time
import sys

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.commands.infer import build_agent_from_config
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common import paths
from deeppavlov.metrics.registry import get_metrics_by_names


# TODO pass paths to local model configs to agent config.


def train_agent_models(config_path: str):
    usr_dir = paths.USR_PATH
    a = build_agent_from_config(config_path)

    for skill_config in a.skill_configs:
        model_config = skill_config['model']
        model_name = model_config['name']

        if issubclass(REGISTRY[model_name], Trainable):
            reader_config = skill_config['dataset_reader']
            reader = from_params(REGISTRY[reader_config['name']], {})
            data = reader.read(reader_config.get('data_path', usr_dir))

            dataset_config = skill_config['dataset']
            dataset_name = dataset_config['name']
            dataset = from_params(REGISTRY[dataset_name], dataset_config, data=data)

            model = from_params(REGISTRY[model_name], model_config)
            model.train(dataset)
        else:
            print('Model {} is not an instance of Trainable, skip training.'.format(model_name),
                  file=sys.stderr)


def train_model_from_config(config_path: str, mode='train'):
    usr_dir = paths.USR_PATH
    config = read_json(config_path)

    reader_config = config['dataset_reader']
    # NOTE: Why there are no params for dataset reader? Because doesn't have __init__()
    reader = from_params(REGISTRY[reader_config['name']], {})
    data = reader.read(reader_config.get('data_path', usr_dir))

    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset = from_params(REGISTRY[dataset_name], dataset_config, data=data)

    vocabs = {}
    if 'vocabs' in config:
        for vocab_param_name, vocab_config in config['vocabs'].items():
            vocab_name = vocab_config['name']
            v = from_params(REGISTRY[vocab_name], vocab_config, mode=mode)
            v.train(dataset.iter_all('train'))
            vocabs[vocab_param_name] = v

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(REGISTRY[model_name], model_config, vocabs=vocabs, mode=mode)

    model.train(dataset)

    # The result is a saved to user_dir trained model.


def train_batches(config_path: str):
    usr_dir = paths.USR_PATH
    config = read_json(config_path)

    train_config = {
        'epochs': 0,
        'batch_size': 1,

        'metrics': ['accuracy'],
        'metric_optimization': 'maximize',

        'validation_patience': 5,
        'val_every_n_epochs': 0,

        'log_every_n_batches': 0,
        'show_examples': False,

        'validate_best': True,
        'test_best': True
    }

    try:
        train_config.update(config['train'])
    except KeyError:
        raise RuntimeError('training config is missing')

    metrics_functions = get_metrics_by_names(train_config['metrics'])

    reader_config = config['dataset_reader']
    reader = from_params(REGISTRY[reader_config['name']], {})
    data = reader.read(reader_config.get('data_path', usr_dir))

    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset: Dataset = from_params(REGISTRY[dataset_name], dataset_config, data=data)

    vocabs = {}
    for vocab_param_name, vocab_config in config.get('vocabs', {}).items():
        vocab_name: Trainable = vocab_config['name']
        v = from_params(REGISTRY[vocab_name], vocab_config)
        v.train(dataset.iter_all('train'))
        v.save()
        vocabs[vocab_param_name] = v

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(REGISTRY[model_name], model_config, vocabs=vocabs, mode='train')

    i = 0
    epochs = 0
    examples = 0
    saved = False
    patience = 0
    best = 0
    log_on = train_config['log_every_n_batches'] > 0
    train_y_true = []
    train_y_predicted = []
    start_time = time.time()
    try:
        while True:
            for batch in dataset.batch_generator(train_config['batch_size']):
                x, y_true = batch
                if log_on:
                    y_predicted = list(model.infer(x))
                    train_y_true += y_true
                    train_y_predicted += y_predicted
                model.train_on_batch(batch)
                i += 1
                examples += len(x)

                if train_config['log_every_n_batches'] > 0 and i % train_config['log_every_n_batches'] == 0:
                    metrics = [f(train_y_true, train_y_predicted) for f in metrics_functions]
                    report = {
                        'epochs_done': epochs,
                        'batches_seen': i,
                        'examples_seen': examples,
                        'metrics': dict(zip(train_config['metrics'], metrics)),
                        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
                    }
                    print('train: {}'.format(report))
                    train_y_true = []
                    train_y_predicted = []

                    if train_config['show_examples']:
                        for xi, ypi, yti in zip(x, y_predicted, y_true):
                            print({'in': xi, 'out': ypi, 'expected': yti})

            epochs += 1

            if train_config['val_every_n_epochs'] > 0 and epochs % train_config['val_every_n_epochs'] == 0:
                val_y_true = []
                val_y_predicted = []
                for x, y_true in dataset.batch_generator(train_config['batch_size'], 'valid'):
                    y_predicted = list(model.infer(x))
                    val_y_true += y_true
                    val_y_predicted += y_predicted

                metrics = [f(train_y_true, train_y_predicted) for f in metrics_functions]

                score = metrics[0]
                if score > best:
                    patience = 0
                    print('Improved on the previous best {} of {}'.format(train_config['metrics'][0], best),
                          file=sys.stderr)
                    best = score
                    model.save()
                    saved = True
                else:
                    patience += 1
                    print('Did not improve on the {} of {}'.format(train_config['metrics'][0], best),
                          file=sys.stderr)

                report = {
                    'examples_seen': len(val_y_true),
                    'metrics': dict(zip(train_config['metrics'], metrics)),
                    'impatience': patience,
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
                }
                if train_config['validation_patience'] > 0:
                    report['patience_limit'] = train_config['validation_patience']
                print('valid: {}'.format(report))
                if train_config['show_examples']:
                    for xi, ypi, yti in zip(x, y_predicted, y_true):
                        print({'in': xi, 'out': ypi, 'expected': yti})

                if patience >= train_config['validation_patience'] > 0:
                    print('Ran out of patience', file=sys.stderr)
                    break

            if epochs >= train_config['epochs'] > 0:
                break
    except KeyboardInterrupt:
        print('Stopped training', file=sys.stderr)

    if not saved:
        print('Saving model', file=sys.stderr)
        model.save()

    if train_config['validate_best'] or train_config['test_best']:
        model = from_params(REGISTRY[model_name], model_config, vocabs=vocabs, mode='infer')
        print('Testing the best saved model', file=sys.stderr)

        if train_config['validate_best']:
            start_time = time.time()
            val_y_true = []
            val_y_predicted = []
            for x, y_true in dataset.batch_generator(train_config['batch_size'], 'valid'):
                y_predicted = list(model.infer(x))
                val_y_true += y_true
                val_y_predicted += y_predicted

            metrics = [f(train_y_true, train_y_predicted) for f in metrics_functions]

            report = {
                'examples_seen': len(val_y_true),
                'metrics': dict(zip(train_config['metrics'], metrics)),
                'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
            }
            print('valid: {}'.format(report))

        if train_config['test_best']:
            start_time = time.time()
            val_y_true = []
            val_y_predicted = []
            for x, y_true in dataset.batch_generator(train_config['batch_size'], 'test'):
                y_predicted = list(model.infer(x))
                val_y_true += y_true
                val_y_predicted += y_predicted

            metrics = [f(train_y_true, train_y_predicted) for f in metrics_functions]

            report = {
                'examples_seen': len(val_y_true),
                'metrics': dict(zip(train_config['metrics'], metrics)),
                'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
            }
            print('test: {}'.format(report))
