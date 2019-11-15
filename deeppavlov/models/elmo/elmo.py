# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

# Modifications copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
from logging import getLogger
from typing import Optional, List

import numpy as np
import tensorflow as tf
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.elmo.bilm_model import LanguageModel
from deeppavlov.models.elmo.elmo2tfhub import export2hub
from deeppavlov.models.elmo.train_utils import average_gradients, clip_grads, safely_str2int, dump_weights

log = getLogger(__name__)


@register('elmo_model')
class ELMo(NNModel):
    """
    The :class:`~deeppavlov.models.elmo.elmo.ELMo` is a deep contextualized word representation that models both
    complex characteristics of word use (e.g., syntax and semantics), and how these uses vary across linguistic
    contexts (i.e., to model polysemy).

    You can use this component for LM training, fine tuning, dumping ELMo to a hdf5 file and wrapping it to
    the tensorflow hub.


    Parameters:
        options_json_path: Path to the json configure.
        char_cnn: Options of char_cnn. For example {"activation":"relu","embedding":{"dim":16},
            "filters":[[1,32],[2,32],[3,64],[4,128],[5,256],[6,512],[7,1024]],"max_characters_per_token":50,
            "n_characters":261,"n_highway":2}
        bidirectional: Whether to use bidirectional or not.
        unroll_steps: Number of unrolling steps.
        n_tokens_vocab: A size of a vocabulary.
        lstm: Options of lstm. It is a dict of "cell_clip":int, "dim":int, "n_layers":int, "proj_clip":int, 
            "projection_dim":int, "use_skip_connections":bool
        dropout: Probability of keeping the network state, values from 0 to 1. 
        n_negative_samples_batch: Whether to use negative samples batch or not. Number of batch samples.
        all_clip_norm_val: Clip the gradients.
        initial_accumulator_value: Whether to use dropout between layers or not.
        learning_rate: Learning rate to use during the training (usually from 0.1 to 0.0001)
        n_gpus: Number of gpu to use.
        seed: Random seed.
        batch_size: A size of a train batch.
        load_epoch_num: An index of loading epoch.
        epoch_load_path: An epoch loading path relative to save_path.
        epoch_save_path:  An epoch saving path relative to save_path.
            If epoch_save_path is None then epoch_save_path = epoch_load_path.
        dumps_save_path: A dump saving path relative to save_path.
        tf_hub_save_path: A tf_hub saving path relative to save_path.

    To train ELMo representations from a paper `Deep contextualized word representations
    <https://arxiv.org/abs/1802.05365>`__ you can use multiple GPUs by set ``n_gpus`` parameter.

    You can explicitly specify the path to a json file with hyperparameters of ELMo used to train by
    ``options_json_path`` parameter.
    The json file must be the same as the json file from `original ELMo implementation
    <https://github.com/allenai/bilm-tf>`__. You can define the architecture using the separate parameters.

    Saving the model will take place in directories with some structure, see below example:

    {MODELS_PATH}/
        elmo_model/
            saves/
                epochs/
                    1/, 2/, .... # directories of epochs
                dumps/
                    weights_epoch_n_1.hdf5, weights_epoch_n_2.hdf5, .... # hdf5 files of dumped ELMo weights
                hubs/
                    tf_hub_model_epoch_n_1/, tf_hub_model_epoch_n_2/, .... # directories of tensorflow hub wrapped
                    ELMo

    Intermediate checkpoints saved to `saves` directory.
    To specify load/save paths use ``load_epoch_num``, ``epoch_load_path``, ``epoch_save_path``, ``dumps_save_path``,
    ``tf_hub_save_path``.

    Dumping and tf_hub wrapping of ELMo occurs after each epoch.

    For learning the LM model dataset like 1 Billion Word Benchmark dataset is needed.
    Examples of how datasets should look like you can learn from the configs of the examples below.

    Vocabulary file is a text file, with one token per line, separated by newlines.
    Each token in the vocabulary is cached as the appropriate 50 character id sequence once.
    It is recommended to always include the special <S> and </S> tokens (case sensitive) in the vocabulary file.

    For fine-tuning of LM on specific data, it is enough to save base model to path
    ``{MODELS_PATH}/elmo_model/saves/epochs/0/`` and start training.

    Also for fine-tuning of LM on specific data, you can use pre-trained model for russian language on different
    datasets.


    LM model pre-trained on `ru-news` dataset ( lines = 63M, tokens = 946M, size = 12GB ), model is available by
    :config:`elmo_lm_ready4fine_tuning_ru_news </elmo/elmo_lm_ready4fine_tuning_ru_news.json>` configuration file
    or :config:`elmo_lm_ready4fine_tuning_ru_news_simple </elmo/elmo_lm_ready4fine_tuning_ru_news_simple.json>`
    configuration file.

    LM model pre-trained on `ru-twitter` dataset ( lines = 104M, tokens = 810M, size = 8.5GB ), model is available by
    :config:`elmo_lm_ready4fine_tuning_ru_twitter </elmo/elmo_lm_ready4fine_tuning_ru_twitter.json>` configuration file
    or :config:`elmo_lm_ready4fine_tuning_ru_twitter_simple </elmo/elmo_lm_ready4fine_tuning_ru_twitter_simple.json>`
    configuration file.

    LM model pre-trained on `ru-wiki` dataset ( lines = 1M, tokens = 386M, size = 5GB ), model is available by
    :config:`elmo_lm_ready4fine_tuning_ru_wiki </elmo/elmo_lm_ready4fine_tuning_ru_wiki.json>` configuration file
    or :config:`elmo_lm_ready4fine_tuning_ru_wiki_simple </elmo/elmo_lm_ready4fine_tuning_ru_wiki_simple.json>`
    configuration file.

    `simple` configuration file is a configuration of a model without special tags of output
    vocab used for first training.

    .. note::

        You need to download about **4 GB** also by default about **32 GB** of RAM and **10 GB** of GPU memory
        required to running the :config:`elmo_lm_ready4fine_tuning_ru_* </elmo/>`
        on one GPU.

    After training you can use ``{MODELS_PATH}/elmo_model/saves/hubs/tf_hub_model_epoch_n_*/``
    as a ``ModuleSpec`` by using `TensorFlow Hub <https://www.tensorflow.org/hub/overview>`__ or by
    DeepPavlov :class:`~deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder`.

    More about the ELMo model you can get from `original ELMo implementation
    <https://github.com/allenai/bilm-tf>`__.


    If some required packages are missing, install all the requirements by running in command line:

    .. code:: bash

        python -m deeppavlov install <path_to_config>

    where ``<path_to_config>`` is a path to one of the :config:`provided config files <elmo_embedder>`
    or its name without an extension, for example :

    .. code:: bash

        python -m deeppavlov install elmo_1b_benchmark_test
        
    Examples:
        For a quick start, you can run test training of the test model on small data by this command from bash:

        .. code:: bash

            python -m deeppavlov train deeppavlov/configs/elmo/elmo_1b_benchmark_test.json -d

        To download the prepared `1 Billion Word Benchmark dataset <http://www.statmt.org/lm-benchmark/>`__ and
        start a training model use this command from bash:

        .. note::

            You need to download about **2 GB** also by default about **10 GB** of RAM and **10 GB** of GPU memory
            required to running :config:`elmo_1b_benchmark <elmo/elmo_1b_benchmark.json>` on one GPU.

        .. code:: bash

            python -m deeppavlov train deeppavlov/configs/elmo/elmo_1b_benchmark.json -d

        To fine-tune ELMo as LM model on `1 Billion Word Benchmark dataset <http://www.statmt.org/lm-benchmark/>`__
        use commands from bash :

        .. code:: bash

            # download the prepared 1 Billion Word Benchmark dataset
            python -m deeppavlov download deeppavlov/configs/elmo/elmo_1b_benchmark.json
            # copy model checkpoint, network configuration, vocabulary of pre-trained LM model
            mkdir -p ${MODELS_PATH}/elmo-1b-benchmark/saves/epochs/0
            cp my_ckpt.data-00000-of-00001 ${MODELS_PATH}/elmo-1b-benchmark/saves/epochs/0/model.data-00000-of-00001
            cp my_ckpt.index ${MODELS_PATH}/elmo-1b-benchmark/saves/epochs/0/model.index
            cp my_ckpt.meta ${MODELS_PATH}/elmo-1b-benchmark/saves/epochs/0/model.meta
            cp checkpoint ${MODELS_PATH}/elmo-1b-benchmark/saves/epochs/0/checkpoint
            cp my_options.json ${MODELS_PATH}/elmo-1b-benchmark/options.json
            cp my_vocab {MODELS_PATH}/elmo-1b-benchmark/vocab-2016-09-10.txt
            # start a fine-tuning
            python -m deeppavlov train deeppavlov/configs/elmo/elmo_1b_benchmark.json

        After training you can use the ELMo model from tf_hub wrapper by
        `TensorFlow Hub <https://www.tensorflow.org/hub/overview>`__ or by
        DeepPavlov :class:`~deeppavlov.models.embedders.elmo_embedder.ELMoEmbedder`:

        >>> from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
        >>> spec = f"{MODELS_PATH}/elmo-1b-benchmark_test/saves/hubs/tf_hub_model_epoch_n_1/"
        >>> elmo = ELMoEmbedder(spec)
        >>> elmo([['вопрос', 'жизни', 'Вселенной', 'и', 'вообще', 'всего'], ['42']])
        array([[ 0.00719104,  0.08544601, -0.07179783, ...,  0.10879009,
                -0.18630421, -0.2189409 ],
            [ 0.16325025, -0.04736076,  0.12354863, ..., -0.1889013 ,
                0.04972512,  0.83029324]], dtype=float32)

    """

    def __init__(self,
                 options_json_path: Optional[str] = None,  # Configure by json file
                 char_cnn: Optional[dict] = None,  # Net architecture by direct params, use for overwrite a json arch.
                 bidirectional: Optional[bool] = None,
                 unroll_steps: Optional[int] = None,
                 n_tokens_vocab: Optional[int] = None,
                 lstm: Optional[dict] = None,
                 dropout: Optional[float] = None,  # Regularization
                 n_negative_samples_batch: Optional[int] = None,  # Train options
                 all_clip_norm_val: Optional[float] = None,
                 initial_accumulator_value: float = 1.0,
                 learning_rate: float = 2e-1,  # For AdagradOptimizer
                 n_gpus: int = 1,  # TODO: Add cpu supporting
                 seed: Optional[int] = None,  # Other
                 batch_size: int = 128,  # Data params
                 load_epoch_num: Optional[int] = None,
                 epoch_load_path: str = 'epochs',
                 epoch_save_path: Optional[str] = None,
                 dumps_save_path: str = 'dumps',
                 tf_hub_save_path: str = 'hubs',
                 **kwargs) -> None:

        # ================ Checking input args =================
        if not (options_json_path or (char_cnn and bidirectional and unroll_steps
                                      and n_tokens_vocab and lstm and dropout and
                                      n_negative_samples_batch and all_clip_norm_val
        )):
            raise Warning('Use options_json_path or/and direct params to set net architecture.')
        self.options = self._load_options(options_json_path)
        self._update_arch_options(char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm)
        self._update_other_options(dropout, n_negative_samples_batch, all_clip_norm_val)

        # Special options
        self.options['learning_rate'] = learning_rate
        self.options['initial_accumulator_value'] = initial_accumulator_value
        self.options['seed'] = seed
        self.options['n_gpus'] = n_gpus
        self.options['batch_size'] = batch_size

        self.permanent_options = self.options

        self.train_options = {}
        self.valid_options = {'batch_size': 256, 'unroll_steps': 1, 'n_gpus': 1}
        self.model_mode = ''

        tf.set_random_seed(seed)
        np.random.seed(seed)

        super().__init__(**kwargs)

        self.epoch_load_path = epoch_load_path

        if load_epoch_num is None:
            load_epoch_num = self._get_epoch_from(self.epoch_load_path, None)

        if epoch_save_path is None:
            self.epoch_save_path = self.epoch_load_path

        self.save_epoch_num = self._get_epoch_from(self.epoch_save_path)

        self.dumps_save_path = dumps_save_path
        self.tf_hub_save_path = tf_hub_save_path

        self._build_model(train=False, epoch=load_epoch_num)

        self.save()
        # after building the model and saving to the specified save path
        # change the way to load intermediate checkpoints
        self.load_path = self.save_path

    def _load_options(self, options_json_path):
        if options_json_path:
            options_json_path = expand_path(options_json_path)
            with open(options_json_path, 'r') as fin:
                options = json.load(fin)
        else:
            options = {}
        return options

    def _update_arch_options(self, char_cnn, bidirectional, unroll_steps, n_tokens_vocab, lstm):
        if char_cnn is not None:
            self.options['char_cnn'] = char_cnn
        if bidirectional is not None:
            self.options['bidirectional'] = bidirectional
        if unroll_steps is not None:
            self.options['unroll_steps'] = unroll_steps
        if n_tokens_vocab is not None:
            self.options['n_tokens_vocab'] = n_tokens_vocab
        if lstm is not None:
            self.options['lstm'] = lstm

    def _update_other_options(self, dropout, n_negative_samples_batch, all_clip_norm_val):
        if dropout is not None:
            self.options['dropout'] = dropout
        if n_negative_samples_batch is not None:
            self.options['n_negative_samples_batch'] = n_negative_samples_batch
        if all_clip_norm_val is not None:
            self.options['all_clip_norm_val'] = all_clip_norm_val

    def _get_epoch_from(self, epoch_load_path, default=0):
        path = self.load_path
        path = path.parent / epoch_load_path
        candidates = path.resolve().glob('[0-9]*')
        candidates = list(safely_str2int(i.parts[-1]) for i in candidates
                          if safely_str2int(i.parts[-1]) is not None)
        epoch_num = max(candidates, default=default)
        return epoch_num

    def _build_graph(self, graph, train=True):
        with graph.as_default():
            with tf.device('/cpu:0'):
                init_step = 0
                global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(init_step), trainable=False)
                self.global_step = global_step
                # set up the optimizer
                opt = tf.train.AdagradOptimizer(learning_rate=self.options['learning_rate'],
                                                initial_accumulator_value=1.0)

                # calculate the gradients on each GPU
                tower_grads = []
                models = []
                loss = tf.get_variable(
                    'train_perplexity', [],
                    initializer=tf.constant_initializer(0.0), trainable=False)
                for k in range(self.options['n_gpus']):
                    with tf.device('/gpu:%d' % k):
                        with tf.variable_scope('lm', reuse=k > 0):
                            # calculate the loss for one model replica and get
                            #   lstm states
                            model = LanguageModel(self.options, True)
                            total_train_loss = model.total_train_loss
                            total_eval_loss = model.total_eval_loss
                            models.append(model)
                            # get gradients
                            grads = opt.compute_gradients(
                                tf.reduce_mean(total_train_loss) * self.options['unroll_steps'],
                                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                            )
                            tower_grads.append(grads)
                            # # keep track of loss across all GPUs
                            if train:
                                loss += total_train_loss
                            else:
                                loss += total_eval_loss

                # calculate the mean of each gradient across all GPUs
                grads = average_gradients(tower_grads, self.options['batch_size'], self.options)
                grads, _ = clip_grads(grads, self.options, True, global_step)
                loss = loss / self.options['n_gpus']
                train_op = opt.apply_gradients(grads, global_step=global_step)
        return models, train_op, loss, graph

    def _init_session(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in self.models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in self.options
        if char_inputs:
            max_chars = self.options['char_cnn']['max_characters_per_token']

        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in self.models
            }
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
                for model in self.models
            }

        if self.options['bidirectional']:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in self.models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
                    for model in self.models
                })

        init_state_values = self.sess.run(init_state_tensors, feed_dict=feed_dict)
        return init_state_values, init_state_tensors, final_state_tensors

    def _fill_feed_dict(self,
                        char_ids_batches,
                        reversed_char_ids_batches,
                        token_ids_batches=None,
                        reversed_token_ids_batches=None):
        # init state tensors
        feed_dict = {t: v for t, v in zip(self.init_state_tensors, self.init_state_values)}

        for k, model in enumerate(self.models):
            start = k * self.options['batch_size']
            end = (k + 1) * self.options['batch_size']

            # character inputs
            char_ids = char_ids_batches[start:end]  # get char_ids

            feed_dict[model.tokens_characters] = char_ids

            if self.options['bidirectional']:
                feed_dict[model.tokens_characters_reverse] = \
                    reversed_char_ids_batches[start:end]  # get tokens_characters_reverse

            if token_ids_batches is not None:
                feed_dict[model.next_token_id] = token_ids_batches[start:end]  # get next_token_id
                if self.options['bidirectional']:
                    feed_dict[model.next_token_id_reverse] = \
                        reversed_token_ids_batches[start:end]  # get next_token_id_reverse

        return feed_dict

    def __call__(self, x, y, *args, **kwargs) -> List[float]:
        if len(args) != 0:
            return []
        char_ids_batches, reversed_char_ids_batches = x
        token_ids_batches, reversed_token_ids_batches = y

        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches, token_ids_batches,
                                         reversed_token_ids_batches)

        with self.graph.as_default():
            loss, self.init_state_values = self.sess.run([self.loss, self.final_state_tensors], feed_dict)
        return loss

    @overrides
    def load(self, epoch: Optional[int] = None) -> None:
        """Load model parameters from self.load_path"""
        path = self.load_path
        if epoch is not None:
            path = path.parent / self.epoch_save_path / str(epoch) / path.parts[-1]
            path.resolve()
            log.info(f'[loading {epoch} epoch]')

        # path.parent.mkdir(parents=True, exist_ok=True)
        path = str(path)

        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            log.info(f'[loading model from {path}]')
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, path)
        else:
            log.info(f'[A checkpoint not found in  {path}]')

    @overrides
    def save(self, epoch: Optional[int] = None) -> None:
        """Save model parameters to self.save_path"""
        path = self.save_path
        if epoch is not None:
            path = path.parent / self.epoch_save_path / str(epoch) / path.parts[-1]
            path.resolve()
            log.info(f'[saving {epoch} epoch]')

        path.parent.mkdir(parents=True, exist_ok=True)
        path = str(path)

        log.info(f'[saving model to {path}]')
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path)

    def train_on_batch(self,
                       x_char_ids: list,
                       y_token_ids: list) -> List[float]:
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            x_char_ids:  a batch of char_ids
            y_token_ids: a batch of token_ids

        Returns:
            value of loss function on batch
        """

        char_ids_batches, reversed_char_ids_batches = x_char_ids
        token_ids_batches, reversed_token_ids_batches = y_token_ids

        feed_dict = self._fill_feed_dict(char_ids_batches, reversed_char_ids_batches,
                                         token_ids_batches, reversed_token_ids_batches)

        with self.graph.as_default():
            loss, _, self.init_state_values = self.sess.run([self.loss, self.train_op, self.final_state_tensors],
                                                            feed_dict)

        return np.mean(loss)

    def _build_model(self, train: bool, epoch: Optional[int] = None, **kwargs):

        if hasattr(self, 'sess'):
            self.sess.close()

        self.options = copy.deepcopy(self.permanent_options)

        if train:
            self.options.update(self.train_options)
            self.options.update(kwargs)

            self.models, self.train_op, self.loss, self.graph = self._build_graph(tf.Graph())
        else:
            self.options.update(self.valid_options)
            self.options.update(kwargs)

            self.models, self.train_op, self.loss, self.graph = self._build_graph(tf.Graph(),
                                                                                  train=False)

        with self.graph.as_default():
            self.init_state_values, self.init_state_tensors, self.final_state_tensors = \
                self._init_session()
        self.load(epoch)

    def process_event(self, event_name, data):
        if event_name == 'before_train' and self.model_mode != 'train':
            self._build_model(train=True)
            self.model_mode = 'train'
        elif event_name == 'before_validation' and self.model_mode != 'validation':
            epoch = self.save_epoch_num + int(data['epochs_done'])
            self.save(epoch)
            self.save()
            self.elmo_export(epoch)

            self._build_model(train=False)
            self.model_mode = 'validation'

    def elmo_export(self, epoch: Optional[int] = None) -> None:
        """
        Dump the trained weights from a model to a HDF5 file and export a TF-Hub module.
        """
        if hasattr(self, 'sess'):
            self.sess.close()
        path = self.save_path
        if epoch:
            from_path = path.parent / self.epoch_save_path / str(epoch) / path.parts[-1]
            weights_to_path = path.parent / self.dumps_save_path / f'weights_epoch_n_{epoch}.hdf5'
            tf_hub_to_path = path.parent / self.tf_hub_save_path / f'tf_hub_model_epoch_n_{epoch}'
            from_path.resolve()
            weights_to_path.resolve()
            tf_hub_to_path.resolve()
            log.info(f'[exporting {epoch} epoch]')
        else:
            from_path = path
            weights_to_path = path.parent / self.dumps_save_path / 'weights.hdf5'
            tf_hub_to_path = path.parent / self.tf_hub_save_path / 'tf_hub_model'

        weights_to_path.parent.mkdir(parents=True, exist_ok=True)
        tf_hub_to_path.parent.mkdir(parents=True, exist_ok=True)

        # Check presence of the model files
        if tf.train.checkpoint_exists(str(from_path)):
            dump_weights(from_path.parent, weights_to_path, self.permanent_options)

            options = copy.deepcopy(self.permanent_options)
            options['char_cnn']['n_characters'] = 262
            export2hub(weights_to_path, tf_hub_to_path, options)

    def destroy(self) -> None:
        """
        Delete model from memory

        Returns:
            None
        """
        if hasattr(self, 'sess'):
            for k in list(self.sess.graph.get_all_collection_keys()):
                self.sess.graph.clear_collection(k)
        super().destroy()
