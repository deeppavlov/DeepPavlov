from typing import Union

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.models.estimator import Estimator


def fit_chainer(config: dict, iterator: Union[DataLearningIterator, DataFittingIterator]):

    chainer_config: dict = config['chainer']
    chainer = Chainer(chainer_config['in'], chainer_config['out'], chainer_config.get('in_y'))
    for component_config in chainer_config['pipe']:
        component = from_params(component_config, vocabs=[], mode='train')
        if 'fit_on' in component_config:
            component: Estimator

            preprocessed = chainer(*iterator.get_instances('train'), to_return=component_config['fit_on'])
            if len(component_config['fit_on']) == 1:
                preprocessed = [preprocessed]
            else:
                preprocessed = zip(*preprocessed)
            component.fit(*preprocessed)
            component.save()

        if 'fit_on_batch' in component_config:
            component: Estimator
            component.fit_batches(iterator, config['train']['batch_size'])
            component.save()

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            chainer.append(component, c_in, c_out, in_y, main)
    return chainer
