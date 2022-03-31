# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import re
from collections import OrderedDict
from logging import getLogger
from operator import itemgetter
from typing import List, Dict, Union

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.bert.bert_classifier import BertClassifierModel

logger = getLogger(__name__)


@register('bert_ranker')
class BertRankerModel(BertClassifierModel):
    """BERT-based model for interaction-based text ranking.

    Linear transformation is trained over the BERT pooled output from [CLS] token.
    Predicted probabilities of classes are used as a similarity measure for ranking.

    Args:
        bert_config_file: path to Bert configuration file
        n_classes: number of classes
        keep_prob: dropout keep_prob for non-Bert layers
        return_probas: set True if class probabilities are returned instead of the most probable label
    """

    def __init__(self, bert_config_file, n_classes=2, keep_prob=0.9, return_probas=True, **kwargs) -> None:
        super().__init__(bert_config_file=bert_config_file, n_classes=n_classes,
                         keep_prob=keep_prob, return_probas=return_probas, **kwargs)

    def train_on_batch(self, features_li: List[List[InputFeatures]], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train the model on the given batch.

        Args:
            features_li: list with the single element containing the batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning rate values
        """

        features = features_li[0]
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return {'loss': loss, 'learning_rate': feed_dict[self.learning_rate_ph]}

    def __call__(self, features_li: List[List[InputFeatures]]) -> Union[List[int], List[List[float]]]:
        """Calculate scores for the given context over candidate responses.

        Args:
            features_li: list of elements where each element contains the batch of features
             for contexts with particular response candidates

        Returns:
            predicted scores for contexts over response candidates
        """

        if len(features_li) == 1 and len(features_li[0]) == 1:
            msg = "It is not intended to use the {} in the interact mode.".format(self.__class__)
            logger.error(msg)
            return [msg]

        predictions = []
        for features in features_li:
            input_ids = [f.input_ids for f in features]
            input_masks = [f.input_mask for f in features]
            input_type_ids = [f.input_type_ids for f in features]

            feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
            if not self.return_probas:
                pred = self.sess.run(self.y_predictions, feed_dict=feed_dict)
            else:
                pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
            predictions.append(pred[:, 1])
        if len(features_li) == 1:
            predictions = predictions[0]
        else:
            predictions = np.hstack([np.expand_dims(el, 1) for el in predictions])
        return predictions
