# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.bert.bert_sequence_tagger import token_from_subtoken

log = getLogger(__name__)


from deeppavlov.models.kbqa.debug_helpers import recursive_shape, recursive_type


@register('mt_bert')
class MultiTaskBert(LRScheduledTFModel):
    """The component for multi task BERT. It builds the BERT body, initiates building of bert heads. 
    When the component `__call__` or `train_on_batch` methods are called the component calls appropriate methods
    of head components. 

    The aggregates components aggreates commponents implementing Bert heads. The head components are called tasks.
    Tasks are split in groups called launches. If tasks they can be in one launch. If tasks are in same launch the
    `__call__` method runs the tasks simultaneously with one call of `tf.Session.run()`.

    Args:
        bert_config_file: path to Bert configuration file
        pretrained_bert: pretrained Bert checkpoint
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        body_learning_rate: learning rate of BERT body
        min_body_learning_rate: min value of body learning rate if learning rate decay is used
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
        freeze_embeddings: set False to train input embeddings
        inference_launch_names: list of names of launches used in the inference mode. If this parameter is provided 
            model is used in inference mode.
    """
    def __init__(self,
                 bert_config_file: str,
                 launches_params: dict,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 body_learning_rate: float = 1e-3,
                 min_body_learning_rate: float = 1e-7,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 freeze_embeddings: bool = True,
                 inference_launch_names: List[str] = None,
                 **kwargs) -> None:
        super().__init__(learning_rate=body_learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)
        self.optimizer_params = {
            "optimizer": optimizer,
            "body_learning_rate": body_learning_rate,
            "min_body_learning_rate": min_body_learning_rate,
            "weight_decay_rate": weight_decay_rate
        }
        self.freeze_embeddings = freeze_embeddings
        self.launches_tasks = launches_params
        self.inference_launch_names = inference_launch_names
        self.mode = 'train' if self.inference_launch_names is None else 'inference'

        self.shared_ph = None

        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        if attention_probs_keep_prob is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 -attention_probs_keep_prob
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_bert_body_graph()
        self.build_tasks()

        self.sess.run(tf.global_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))
            if tf.train.checkpoint_exists(pretrained_bert) \
                    and not (self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve()))) \
                    and self.mode == 'train':
                log.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                task_names = []
                for launch_params in self.launches_tasks.values():
                    for task_obj in launch_params['tasks'].values():
                        task_names.append(task_obj.task_name)
                var_list = self._get_saveable_variables(
                    exclude_scopes=('Optimizer', 'learning_rate', 'momentum') + tuple(task_names))
                saver = tf.train.Saver(var_list)
                saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

    def build_tasks(self):
        def get_train_op(*args, **kwargs):
            return self.get_train_op(*args, **kwargs)
        for launch_name, launch_params in self.launches_tasks.items():
            for task_name, task_obj in launch_params['tasks'].items():
                task_obj.build(
                    bert_body=self.bert,
                    optimizer_params=self.optimizer_params,
                    shared_placeholders=self.shared_ph,
                    sess=self.sess,
                    mode=self.mode,
                    get_train_op_func=get_train_op,
                    freeze_embeddings=self.freeze_embeddings
                )

    def _init_shared_placeholders(self) -> None:
        self.shared_ph = {
            'input_ids': tf.placeholder(shape=(None, None),
                                        dtype=tf.int32,
                                        name='token_indices_ph'),
            'input_masks': tf.placeholder(shape=(None, None),
                                          dtype=tf.int32,
                                          name='token_mask_ph'),
            'learning_rate': tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph'),
            'keep_prob': tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph'),
            'is_train': tf.placeholder_with_default(False, shape=[], name='is_train_ph')}
        self.shared_ph['token_types'] = tf.placeholder_with_default(
                tf.zeros_like(self.shared_ph['input_ids'], dtype=tf.int32),
                shape=self.shared_ph['input_ids'].shape,
                name='token_types_ph')

    def _init_bert_body_graph(self) -> None:
        self._init_shared_placeholders()
        sph = self.shared_ph
        self.bert = BertModel(config=self.bert_config,
                              is_training=sph['is_train'],
                              input_ids=sph['input_ids'],
                              input_mask=sph['input_masks'],
                              token_type_ids=sph['token_types'],
                              use_one_hot_embeddings=False)

    def save(self, exclude_scopes=('Optimizer', 'learning_rate', 'momentum')) -> None:
        return super().save(exclude_scopes=exclude_scopes)

    def load(self,
             exclude_scopes=('Optimizer',
                             'learning_rate',
                             'momentum'),
             **kwargs) -> None:
        return super().load(exclude_scopes=exclude_scopes, **kwargs)

    def train_on_batch(self, *args, **kwargs) -> None:
        """Calls `train_on_batch` methods for every task. This method takes either `args` or `kwargs` not both.
        The order of `args` is the same as the order of tasks in the component parameters:
        ```python
        args = [
            launch1_task1_in_x[0],
            launch1_task1_in_x[1],
            launch1_task1_in_x[2],
            ...
            launch1_task1_in_y[0],
            launch1_task1_in_y[1],
            ...
            launch1_task2_in_x[0],
            ...
            launch2_task1_in_x[0],
            ...
        ]
        ```
        
        If `kwargs` are used keys of args have to match values of `in_names` and `in_y_names` params passed to the
        tasks. 
        """
        # TODO: test passing arguments as args
        if args and kwargs:
            raise ValueError("You can use either args or kwargs not both")
        tasks = []
        for launch_params in self.launches_tasks.values():
            for task in launch_params['tasks'].values():
                tasks.append(task)
        num_x_args = sum([len(task.in_names) for task in tasks])
        num_used_x_args, num_used_y_args = 0, num_x_args
        for task in tasks:
            if args:
                kw = {
                    inp_name: a for inp_name, a 
                    in zip(
                        task.in_names + task.in_y_names, 
                        args[num_used_x_args:num_used_x_args+len(task.in_names)] 
                            + args[num_used_y_args:num_used_y_args+len(task.in_y_names)],
                    )
                }
                num_used_x_args += len(task.in_names)
                num_used_y_args += len(task.in_y_names)
            else:
                kw = {inp_name: kwargs[inp_name] for inp_name in task.in_names + task.in_y_names}
            task.train_on_batch(
                **kw, 
                body_learning_rate=max(self.get_learning_rate(), self.optimizer_params['min_body_learning_rate'])
            )

    def __call__(self, *args, launch_name: Optional[str] = None, **kwargs):
        """Calls one or several BERT heads depending on provided launch names. `args` and `kwargs` contain 
        inputs of BERT tasks. `args` and `kwargs cannot be used simultaneously. If `args` are used `args` content has 
        to be
        ```python
        args = [
            launch1_task1_in_x[0],
            launch1_task1_in_x[1],
            ...
            launch1_task2_in_x[0],
            launch1_task2_in_x[1],
            ...
            launch2_task1_in_x[0],
            launch2_task1_in_x[1],
            ...
        ]
        ```

        If `kwargs` is used `kwargs` keys has to match content of `in_names` params of called tasks.

        Args:
            launch_name: launch which which task are called. If `launch_name` is not provided and 
                `self.inference_launches` is not `None` launches from `self.inference_launches` are run. If 
                `launch_name` is `None` and `self.inference_launches` is `None` all launches are run.
        
        Returns:
            list results of called tasks.
        """
        if args and kwargs:
            raise ValueError("You may use either args or kwargs not both")
        if launch_name is None:
            if self.inference_launch_names is None:
                launch_names = list(self.launches_tasks.keys())
            else:
                launch_names = self.inference_launch_names
        else:
            launch_names = [launch_name]
        results = []
        for launch_name in launch_names:
            fetches = []
            feed_dict = {}
            tasks = []
            for launch_tasks in self.launches_tasks[launch_name].values():
                for task in launch_tasks.values():
                    tasks.append(task)
            num_used_args = 0
            for task in tasks:
                if args:
                    kw = {
                        inp_name: a for inp_name, a 
                        in zip(task.in_names, args[num_used_args:num_used_args+len(task.in_names)])
                    }
                    num_used_args += len(task.in_names)
                else:
                    kw = {inp_name: kwargs[inp_name] for inp_name in task.in_names}
                task_fetches, task_feed_dict = task.get_sess_run_infer_args(**kw)
                fetches.append(task_fetches)
                feed_dict.update(task_feed_dict)
            sess_run_res = self.sess.run(fetches, feed_dict=feed_dict)
            for task, srs in zip(tasks, sess_run_res):
                task_results = task.post_process_preds(srs)
                results.append(task_results)
        return results


class MTBertTask(ABC):
    """Abstract class for multitask Bert tasks. Objects of its subclasses are linked with bert body when `build` 
    method is called. Training is performed with `train_on_batch` method. One training iteration consists of one
    `train_on_batch` consequent call for every task. The objects of classes derived from `MTBertTask` don't have 
    `__call__` method. Instead they have `get_sess_run_infer_args` and `post_process_preds` which are called from 
    object of `MultiTaskBert` class.

    Args:
        task_name: the name of the multitask Bert task used as variable scope. This parameter should have equal
            values in training and inference configs.
        keep_prob: dropout keep_prob for non-Bert layers 
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        learning_rate: learning rate of BERT head
        in_names: list of inputs of the model. These should be a subset of "in" list of "multitask_bert" component
            in the config.
        in_y_names: list of y inputs of the model. These should be a subset of "in_y" list of "multitask_bert"
            component in the config.
    """
    def __init__(
            self,
            task_name: str = "seq_tagging",
            keep_prob: float = 1.,
            return_probas: bool = None,
            learning_rate: float = 1e-3,
            in_names: List[str] = None,
            in_y_names: List[str] = None,
    ):
        self.task_name = task_name
        self.keep_prob = keep_prob
        self.return_probas = return_probas
        self.init_head_learning_rate = learning_rate
        self.min_body_learning_rate = None
        self.head_learning_rate_multiplier = None
        self.in_names = in_names
        self.in_y_names = in_y_names

        self.bert = None
        self.optimizer_params = None
        self.shared_ph = None
        self.shared_feed_dict = None
        self.sess = None
        self.get_train_op_func = None
        self.freeze_embeddings = None

    def build(
            self, 
            bert_body: BertModel, 
            optimizer_params: Dict[str, Union[str, float]], 
            shared_placeholders: Dict[str, tf.placeholder], 
            sess: tf.Session, 
            mode: str, 
            get_train_op_func: Callable,
            freeze_embeddings: bool) -> None:
        """Initiates building of the Bert head and initializes optimizer parameters, placeholders that are common for 
        all tasks.

        Args:
            bert_body: instance of `BertModel`.
            optimizer_params: a dictionary with four fields: 
                'optimizer' -- a name of optimizer class,
                'body_learning_rate' (float) -- initial value of Bert body learning rate,
                'min_body_learning_rate' (float) -- min Bert body learning rate for learning rate decay,
                'weight_decay_rate' (float) -- L2 weight decay for `AdamWeightDecayOptimizer`
            shared_placeholders: a dictionary with placeholders used in all tasks. The dictionary contains fields
                 'input_ids', 'input_masks', 'learning_rate', 'keep_prob', 'is_train', 'token_types'.
            sess: current `tf.Session` instance
            mode: 'train' or 'inference'
            get_train_op_func: a function returning `tf.Operation` and with signature similar to 
                `LRScheduledTFModel.get_train_op` without `self` argument.
            freeze_embeddings: set False to train input embeddings
        """
        self.get_train_op_func = get_train_op_func
        self.freeze_embeddings = freeze_embeddings
        self.bert = bert_body
        self.optimizer_params = optimizer_params
        if mode == 'train':
            self.head_learning_rate_multiplier = \
                self.init_head_learning_rate / self.optimizer_params['body_learning_rate']
        else:
            self.head_learning_rate_multiplier = 0
        mblr = self.optimizer_params.get('min_body_learning_rate')
        self.min_body_learning_rate = 0. if mblr is None else mblr
        self.shared_ph = shared_placeholders
        self.sess = sess
        self._init_graph()
        if mode == 'train':
            self._init_optimizer()

    @abstractmethod
    def _init_graph(self) -> None:
        """Build Bert head, initialize task specific placeholders, create attributes containing output probabilities
        and model loss. Optimizer initialized not in this method but in `_init_optimizer`."""
        pass

    def get_train_op(self, loss: tf.Tensor, body_learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        """Return operation for the task training. Head learning rate is calculated as a product of 
        `body_learning_rate` and quotient of initial head learning rate and body learning rate.

        Args:
            loss: the task loss
            body_learning_rate: the learning rate for the Bert body
 
        Returns:
            train operation for the task
        """
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('bert/encoder', 'bert/embeddings')
        if self.freeze_embeddings:
            kwargs['learnable_scopes'] = ('bert/encoder',)
        learning_rate = body_learning_rate * self.head_learning_rate_multiplier
        bert_train_op = self.get_train_op_func(loss, body_learning_rate, **kwargs)
        # train_op for ner head variables
        kwargs['learnable_scopes'] = (self.task_name,)
        head_train_op = self.get_train_op_func(loss, learning_rate, **kwargs)
        return tf.group(bert_train_op, head_train_op)

    def _init_optimizer(self) -> None:
        with tf.variable_scope(self.task_name):
            with tf.variable_scope('Optimizer'):
                self.global_step = tf.get_variable('global_step',
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0),
                                                   trainable=False)
                # default optimizer for Bert is Adam with fixed L2 regularization

            if self.optimizer_params.get('optimizer') is None:
                self.train_op = \
                    self.get_train_op(
                        self.loss,
                        body_learning_rate=self.shared_ph['learning_rate'],
                        optimizer=AdamWeightDecayOptimizer,
                        weight_decay_rate=self.optimizer_params.get('weight_decay_rate', 1e-6),
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-6,
                        optimizer_scope_name='Optimizer',
                        exclude_from_weight_decay=["LayerNorm",
                                                   "layer_norm",
                                                   "bias"])
            else:
                self.train_op = self.get_train_op(self.loss,
                                                  body_learning_rate=self.shared_ph['learning_rate'],
                                                  optimizer_scope_name='Optimizer')

            if self.optimizer_params.get('optimizer') is None:
                with tf.variable_scope('Optimizer'):
                    new_global_step = self.global_step + 1
                    self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    def _build_feed_dict(self, input_ids, input_masks, token_types, y=None, body_learning_rate=None):
        sph = self.shared_ph
        train = y is not None
        feed_dict = {
            sph['input_ids']: input_ids,
            sph['input_masks']: input_masks,
            sph['token_types']: token_types,
            sph['is_train']: train,
        }
        if train:
            feed_dict.update({
                sph['learning_rate']: body_learning_rate,
                self.y_ph: y,
                sph['keep_prob']: self.keep_prob,
            })
        return feed_dict

    def train_on_batch(self, **kwargs) -> Dict[str, float]:
        """Trains the task on one batch. This method will work correctly if you override `get_sess_run_train_args`
        for your task.
        
        Args:
            kwargs: the keys are `body_learning_rate` and contents of `in_names`, `in_y_names` attributes.

        Returns:
            dictionary with loss and body and head learning rates.
        """
        fetches, feed_dict = self.get_sess_run_train_args(**kwargs)
        _, loss = self.sess.run(fetches, feed_dict=feed_dict)
        return {'loss': loss,
                f'{self.task_name}_head_learning_rate': 
                    float(kwargs['body_learning_rate']) * self.head_learning_rate_multiplier,
                'bert_learning_rate': kwargs['body_learning_rate']}

    @abstractmethod
    def get_sess_run_infer_args(self, **kwargs) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for inference. Fetches are lists of tensors and feed_dict is dictionary
        with placeholder values required for fetches computation. The method is used for when `MultiTaskBert` `__call__`
        method is used.

        If `self.return_probas` is True fetches contains probabilities tensor and predictions tensor otherwise.

        Args:
            kwargs: the keys of `kwargs` are model `in_names` attribute content.
        
        Returns:
            fetches and feed_dict
        """
        pass

    @abstractmethod
    def get_sess_run_train_args(self, **kwargs) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for task `train_on_batch` method.

        Args:
            kwargs: the keys of `kwargs` are model `in_names` and `in_y_names` attributes content and 
                'body_learning_rate`.

        Returns:
            fetches and feed_dict
        """
        pass
    
    @abstractmethod
    def post_process_preds(self, sess_run_res: list) -> list:
        """Post process results of `tf.Session.run` called for task inference. Called from method 
        `MultiTaskBert.__call__`.

        Args:
            sess_run_res: computed fetches from get_sess_run_infer_args

        Returns:
            postprocessed results
        """
        pass


@register("mt_bert_seq_tagging_task")
class MTBertSequenceTaggingTask(MTBertTask):
    """BERT head for text tagging. It predicts a label for every token (not subtoken) in the text.
    You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.
    Objects of this class should be passed to the constructor of `MultitaskBert` class in param `launches_params`.

    Args:
        task_name: the name of the multitask Bert task used as variable scope. This parameter should have equal
            values in training and inference configs.
        n_tags: number of distinct tags
        use_crf: whether to use CRF on top or not
        use_birnn: whether to use bidirection rnn after BERT layers.
            For NER and morphological tagging we usually set it to `False` as otherwise the model overfits
        birnn_cell_type: the type of Bidirectional RNN. Either `lstm` or `gru`
        birnn_hidden_size: number of hidden units in the BiRNN layer in each direction
        keep_prob: dropout keep_prob for non-Bert layers 
        encoder_dropout: dropout probability of encoder output layer
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        encoder_layer_ids: list of averaged layers from Bert encoder (layer ids)
            optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
            weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer
        learning_rate: learning rate of BERT head
        in_names: list of inputs of the model. These should be a subset of "in" list of "multitask_bert" component
            in config.
        in_y_names: list of y inputs of the model. These should be a subset of "in_y" list of "multitask_bert"
            component in config.
    """
    def __init__(
            self,
            task_name: str = "seq_tagging",
            n_tags: int = None,
            use_crf: bool = None,
            use_birnn: bool = False,
            birnn_cell_type: str = 'lstm',
            birnn_hidden_size: int = 128,
            keep_prob: float = 1.,
            encoder_dropout: float = 0.,
            return_probas: bool = None,
            encoder_layer_ids: List[int] = None,
            learning_rate: float = 1e-3,
            in_names: List[str] = None,
            in_y_names: List[str] = None,
    ):
        super().__init__(
            task_name, 
            keep_prob, 
            return_probas, 
            learning_rate, 
            in_names, 
            in_y_names)
        self.n_tags = n_tags
        self.use_crf = use_crf
        self.use_birnn = use_birnn
        self.birnn_cell_type = birnn_cell_type
        self.birnn_hidden_size = birnn_hidden_size
        self.encoder_dropout = encoder_dropout
        self.encoder_layer_ids = encoder_layer_ids

    def _init_placeholders(self) -> None:
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None),
                                         dtype=tf.int32,
                                         name='y_mask_ph')
        self.encoder_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='encoder_keep_prob_ph')

    def _init_graph(self) -> None:
        with tf.variable_scope(self.task_name):
            self._init_placeholders()
            self.seq_lengths = tf.reduce_sum(self.y_masks_ph, axis=1)

            layer_weights = tf.get_variable('layer_weights_',
                                            shape=len(self.encoder_layer_ids),
                                            initializer=tf.ones_initializer(),
                                            trainable=True)
            layer_mask = tf.ones_like(layer_weights)
            layer_mask = tf.nn.dropout(layer_mask, self.encoder_keep_prob)
            layer_weights *= layer_mask
            # to prevent zero division
            mask_sum = tf.maximum(tf.reduce_sum(layer_mask), 1.0)
            layer_weights = tf.unstack(layer_weights / mask_sum)
            # TODO: may be stack and reduce_sum is faster
            units = sum(w * l for w, l in zip(layer_weights, self.encoder_layers()))
            units = tf.nn.dropout(units, keep_prob=self.shared_ph['keep_prob'])
            if self.use_birnn:
                units, _ = bi_rnn(units,
                                  self.birnn_hidden_size,
                                  cell_type=self.birnn_cell_type,
                                  seq_lengths=self.seq_lengths,
                                  name='birnn')
                units = tf.concat(units, -1)
            # TODO: maybe add one more layer?
            logits = tf.layers.dense(units, units=self.n_tags, name="output_dense")

            self.logits = token_from_subtoken(logits, self.y_masks_ph)

            # CRF
            if self.use_crf:
                transition_params = tf.get_variable('Transition_Params',
                                                    shape=[self.n_tags, self.n_tags],
                                                    initializer=tf.zeros_initializer())
                log_likelihood, transition_params = \
                    tf.contrib.crf.crf_log_likelihood(self.logits,
                                                      self.y_ph,
                                                      self.seq_lengths,
                                                      transition_params)
                loss_tensor = -log_likelihood
                self._transition_params = transition_params

            self.y_predictions = tf.argmax(self.logits, -1)
            self.y_probas = tf.nn.softmax(self.logits, axis=2)

            with tf.variable_scope("loss"):
                tag_mask = self._get_tag_mask()
                y_mask = tf.cast(tag_mask, tf.float32)
                if self.use_crf:
                    self.loss = tf.reduce_mean(loss_tensor)
                else:
                    self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_ph,
                                                                       logits=self.logits,
                                                                       weights=y_mask)

    def _get_tag_mask(self) -> tf.Tensor:
        """
        Returns: tag_mask,
            a mask that selects positions corresponding to word tokens (not padding and `CLS`)
        """
        max_length = tf.reduce_max(self.seq_lengths)
        one_hot_max_len = tf.one_hot(self.seq_lengths - 1, max_length)
        tag_mask = tf.cumsum(one_hot_max_len[:, ::-1], axis=1)[:, ::-1]
        return tag_mask

    def encoder_layers(self):
        """
        Returns: the output of BERT layers specfied in ``self.encoder_layers_ids``
        """
        return [self.bert.all_encoder_layers[i] for i in self.encoder_layer_ids]

    @overrides
    def _build_feed_dict(self, input_ids, input_masks, y_masks, y=None, body_learning_rate=None):
        token_types = np.zeros(np.array(input_ids).shape)
        sph = self.shared_ph
        train = y is not None
        feed_dict = super()._build_feed_dict(input_ids, input_masks, token_types, y, body_learning_rate)
        if train:
            feed_dict[self.encoder_keep_prob] = 1.0 - self.encoder_dropout
        feed_dict[self.y_masks_ph] = y_masks
        return feed_dict

    def _decode_crf(self, feed_dict: Dict[tf.Tensor, np.ndarray]) -> List[np.ndarray]:
        logits, trans_params, mask, seq_lengths = self.sess.run([self.logits,
                                                                 self._transition_params,
                                                                 self.y_masks_ph,
                                                                 self.seq_lengths],
                                                                feed_dict=feed_dict)
        # iterate over the sentences because no batching in viterbi_decode
        y_pred = []
        for logit, sequence_length in zip(logits, seq_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]
        return y_pred

    def _check_in_names(self):
        if len(self.in_names) != 3:
            raise ValueError(
                f"Sequence tagging task of multitask Bert takes exactly 3 arguments: input ids, input masks "
                "and y masks whereas {len(self.in_names)} arguments {self.in_names} are given.")       

    def _check_in_y_names(self):
        if len(self.in_y_names) != 1:
            raise ValueError(
                f"Sequence tagging task of multitask Bert in_y has to have 1 element "
                "whereas its length is {len(self.in_y_names)}.\nin_y_names = {self.in_y_names}")       

    def get_sess_run_infer_args(self, **kwargs) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model inference. The method is called from `MultiTaskBert.__call__`.

        Args:
            kwargs: a dictionary of size 3 containing input ids, input masks, y masks. The keys should be equal to
                `in_names` attribute.

        Returns:
            list of fetches and feed_dict
        """
        self._check_in_names()
        build_feed_dict_args = [kwargs[k] for k in self.in_names]
        feed_dict = self._build_feed_dict(*build_feed_dict_args)
        if self.return_probas:
            fetches = self.y_probas
        else:
            if self.use_crf:
                fetches = [self.logits, self._transition_params, self.y_masks_ph, self.seq_lengths]
            else:
                fetches = [self.y_predictions, self.seq_lengths]
        return fetches, feed_dict

    def get_sess_run_train_args(self, **kwargs) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model `train_on_batch` method.

        Args:
            kwargs: a dictionary which keys are elements of `in_names` attribute, `in_y_names` attribute and
                `'body_learning_rate'`.

        Returns:
            list of fetches and feed_dict
        """
        self._check_in_names()
        self._check_in_y_names()
        build_feed_dict_args = [kwargs[k] for k in self.in_names]
        y = kwargs[self.in_y_names[0]]
        lr = kwargs['body_learning_rate']
        feed_dict = self._build_feed_dict(*build_feed_dict_args, y=y, body_learning_rate=lr)
        fetches = [self.train_op, self.loss]
        return fetches, feed_dict

    def post_process_preds(self, sess_run_res):
        """Decodes CRF if needed and returns predictions or probabilities.

        Args:
            sess_run_res: list of computed fetches gathered by `get_sess_run_infer_args`

        Returns:
            predictions or probabilities depending on `return_probas` attribute
        """
        if self.return_probas:
            pred = sess_run_res
        else:
            if self.use_crf:
                logits, trans_params, mask, seq_lengths = sess_run_res
                pred = []
                for logit, sequence_length in zip(logits, seq_lengths):
                    logit = logit[:int(sequence_length)]  # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                    pred += [viterbi_seq]
            else:
                pred, seq_lengths = sess_run_res
                pred = [p[:l] for l, p in zip(seq_lengths, pred)]
        return pred


@register("mt_bert_classification_task")
class MTBertClassificationTask(MTBertTask):
    """Task for text classification.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        task_name: the name of the multitask Bert task used as variable scope. This parameter should have equal
            values in training and inference configs.
        n_classes: number of classes
        return_probas: set True if return class probabilites instead of most probable label needed
        one_hot_labels: set True if one-hot encoding for labels is used
        keep_prob: dropout keep_prob for non-Bert layers
        multilabel: set True if it is multi-label classification
        learning_rate: learning rate of BERT head
        optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
        in_names: list of inputs of the model. These should be a subset of "in" list of "multitask_bert" component
            in config.
        in_y_names: list of y inputs of the model. These should be a subset of "in_y" list of "multitask_bert"
            component in config.
    """
    def __init__(
            self,
            task_name: str = "classification",
            n_classes: int = None,
            return_probas: bool = None,
            one_hot_labels: bool = None,
            keep_prob: float = 1.,
            multilabel: bool = False,
            learning_rate: float = 2e-5,
            optimizer: str = "Adam",
            in_names: List[str] = None,
            in_y_names: List[str] = None,
    ):
        super().__init__(
            task_name, 
            keep_prob, 
            return_probas, 
            learning_rate, 
            in_names, 
            in_y_names)
        self.n_classes = n_classes
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

    def _init_placeholders(self):
        if not self.one_hot_labels:
            self.y_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32, name='y_ph')

    def _init_graph(self):
        with tf.variable_scope(self.task_name):
            self._init_placeholders()

            output_layer = self.bert.get_pooled_output()
            hidden_size = output_layer.shape[-1].value

            output_weights = tf.get_variable(
                "output_weights", [self.n_classes, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [self.n_classes], initializer=tf.zeros_initializer())

            with tf.variable_scope("loss"):
                output_layer = tf.nn.dropout(output_layer, keep_prob=self.shared_ph['keep_prob'])
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)

                if self.one_hot_labels:
                    one_hot_labels = self.y_ph
                else:
                    one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)

                self.y_predictions = tf.argmax(logits, axis=-1)
                if not self.multilabel:
                    log_probs = tf.nn.log_softmax(logits, axis=-1)
                    self.y_probas = tf.nn.softmax(logits, axis=-1)
                    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                    self.loss = tf.reduce_mean(per_example_loss)
                else:
                    self.y_probas = tf.nn.sigmoid(logits)
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))

    def _check_in_names(self):
        if len(self.in_names) != 1:
            raise ValueError(
                f"Classification task of multitask Bert takes exactly 1 arguments: Bert features "
                "whereas {len(self.in_names)} arguments {self.in_names} are given.")       

    def _check_in_y_names(self):
        if len(self.in_y_names) != 1:
            raise ValueError(
                f"Classification task of multitask Bert in_y has to have 1 element "
                "whereas its length is {len(self.in_y_names)}.\nin_y_names = {self.in_y_names}")       

    def get_sess_run_train_args(self, **kwargs) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model `train_on_batch` method.

        Args:
            kwargs: a dictionary which keys are elements of `in_names` attribute, `in_y_names` attribute and
                `'body_learning_rate'`.

        Returns:
            list of fetches and feed_dict
        """
        self._check_in_names()
        self._check_in_y_names()
        features = kwargs[self.in_names[0]]
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]
        y = kwargs[self.in_y_names[0]]
        lr = kwargs['body_learning_rate']
        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y=y, body_learning_rate=lr)
        fetches = [self.train_op, self.loss]
        return fetches, feed_dict

    def get_sess_run_infer_args(self, **kwargs) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model inference. The method is called from `MultiTaskBert.__call__`.

        Args:
            kwargs: a dictionary of size 1 containing Bert features. The keys should be equal to
                `in_names` attribute.

        Returns:
            list of fetches and feed_dict
        """
        self._check_in_names()
        features = kwargs[self.in_names[0]]
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]
        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
        fetches = self.y_probas if self.return_probas else self.y_predictions
        return fetches, feed_dict

    def post_process_preds(self, sess_run_res):
        """Returns `tf.Session.run` results for inference without changes."""
        return sess_run_res


@register("mt_bert_reuser")
class MTBertReUser:
    """Instances of this class are for calling MultiTaskBert launches. In config pipe `MultiTaskBert` class is 
    declared for inference on 1 launch. To use another launch `MTBertReUser` instance is created.

    Args:
        mt_bert: an instance of `MultiTaskBert`
        launch_name: the name of the launch of in multitask Bert
    """
    def __init__(self, mt_bert: MultiTaskBert, launch_name: str, *args, **kwargs):
        self.mt_bert = mt_bert
        self.launch_name = launch_name

    def __call__(self, *args, **kwargs):
        return self.mt_bert(*args, **kwargs, launch_name=self.launch_name)


@register("input_splitter")
class InputSplitter:
    """The instance of these class in pipe splits a sequence of dictionaries into independent variables.

    Args:
        keys_to_extract: a sequence of strings which have to match keys of dictionary which will be splitted
    """
    def __init__(self, keys_to_extract: Union[List[str], Tuple[str, ...]], **kwargs):
        self.keys_to_extract = keys_to_extract

    def __call__(self, inp: List[dict]) -> List[list]:
        """Returns lists of values of dictionaries from `inp`. Every list contains values which have a key from 
        `keys_to_extract` attribute. The order of elements of `keys_to_extract` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dicionaries from `inp`
        """
        extracted = [[] for _ in self.keys_to_extract]
        for item in inp:
            for i, key in enumerate(self.keys_to_extract):
                extracted[i].append(item[key])
        return extracted

