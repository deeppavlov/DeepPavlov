"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from copy import copy, deepcopy
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Concatenate, Reshape, CuDNNLSTM, Lambda
from keras import backend as K
from overrides import overrides
from pathlib import Path

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.models.classifiers.intents.intent_model import KerasIntentModel
from deeppavlov.models.classifiers.intents import metrics as metrics_file
from deeppavlov.models.classifiers.intents.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.classifiers.intents.utils import md5_hashsum
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.core.common.log import get_logger
from deeppavlov.models.evolution.check_binary_mask import number_to_type_layer, \
    find_sources_and_sinks, get_digraph_from_binary_mask
from deeppavlov.models.evolution.utils import Attention, expand_tile
from deeppavlov.core.common.file import save_json, read_json


log = get_logger(__name__)


@register('evolution_classification_model')
class KerasEvolutionClassificationModel(KerasIntentModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opt["binary_mask"] = np.array(self.opt["binary_mask"])

    def get_node_output(self, node_id, dg, params, edges_outputs=None, inp=None):
        if inp is None:
            input_nodes = [edge[0] for edge in dg.in_edges(node_id)]
            inp_list = []
            for input_node in input_nodes:
                if len(K.int_shape(edges_outputs[input_node])) == 3:
                    inp_list.append(edges_outputs[input_node])
                elif len(K.int_shape(edges_outputs[input_node])) == 2:
                    input_expanded = Lambda(lambda x: expand_tile(x, axis=1))(edges_outputs[input_node])
                    inp_list.append(input_expanded)
                else:
                    raise ValueError("All the layers should take in and take out 2 and 3 dimensional tensors!")
            if len(input_nodes) > 1:
                try:
                    inp = Concatenate()(inp_list)
                except ValueError:
                    time_steps = []
                    features = []
                    for i in range(len(inp_list)):
                        if len(K.int_shape(inp_list[i])) == 2:
                            inp_list[i] = Lambda(lambda x: expand_tile(x, axis=1))(inp_list[i])
                        time_steps.append(K.int_shape(inp_list[i])[1])
                        features.append(K.int_shape(inp_list[i])[2])
                    new_feature_shape = max(features)
                    for i in range(len(inp_list)):
                        inp_list[i] = Dense(new_feature_shape)(inp_list[i])
                    inp = Concatenate(axis=1)(inp_list)
            else:
                inp = inp_list[0]

        if params[params["nodes"][str(node_id)]]["node_name"] == "BiCuDNNLSTM":
            node_params = deepcopy(params[params["nodes"][str(node_id)]])
            node_params.pop("node_name")
            node_params.pop("node_type")
            node_params.pop("node_layer")
            output_of_node = Bidirectional(CuDNNLSTM(**node_params))(inp)
        else:
            node_func = globals().get(params[params["nodes"][str(node_id)]]["node_name"], None)
            node_params = deepcopy(params[params["nodes"][str(node_id)]])
            node_params.pop("node_name")
            node_params.pop("node_type")
            node_params.pop("node_layer")
            if callable(node_func):
                output_of_node = node_func(**node_params)(inp)
            else:
                raise AttributeError("Node {} is not defined correctly".format(node_id))
        return output_of_node

    def evolution_classification_model(self, params):
        """
        Build un-compiled model of shallow-and-wide CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """
        inp = Input(shape=(params['text_size'], params['embedding_size']))

        dg = get_digraph_from_binary_mask(params["nodes"], np.array(params["binary_mask"]))
        sources, sinks, isolates = find_sources_and_sinks(dg)

        edges_outputs = {}

        sequence_of_nodes = []
        sequence_of_nodes.append(sources)

        while True:
            if set(sinks).issubset(set(sum(sequence_of_nodes, []))):
                break
            next_nodes = []
            for node_id in sequence_of_nodes[-1]:
                out_edges = dg.out_edges(node_id)
                for edge in out_edges:
                    in_nodes_to_edge = [in_edge[0] for in_edge in dg.in_edges(edge[1])]
                    if set(in_nodes_to_edge).issubset(set(sum(sequence_of_nodes, []))):
                        next_nodes.append(edge[1])
            sequence_of_nodes.append(next_nodes)

        sequence_of_nodes = sum(sequence_of_nodes, [])

        for node_id in sequence_of_nodes:
            if node_id in sources:
                edges_outputs[node_id] = self.get_node_output(node_id, dg, params, inp=inp)
            elif node_id in isolates:
                pass
            else:
                edges_outputs[node_id] = self.get_node_output(node_id, dg, params, edges_outputs=edges_outputs)

        if len(sinks) == 1:
            output = edges_outputs[sinks[0]]
        else:
            outputs = []
            for sink in sinks:
                outputs.append(edges_outputs[sink])
            try:
                output = Concatenate()(outputs)
            except ValueError:
                time_steps = []
                features = []
                for i in range(len(outputs)):
                    if len(K.int_shape(outputs[i])) == 2:
                        outputs[i] = Lambda(lambda x: expand_tile(x, axis=1))(outputs[i])
                    time_steps.append(K.int_shape(outputs[i])[1])
                    features.append(K.int_shape(outputs[i])[2])
                new_feature_shape = max(features)
                for i in range(len(outputs)):
                    outputs[i] = Dense(new_feature_shape)(outputs[i])
                output = Concatenate(axis=1)(outputs)

        if len(output.shape) == 3:
            output = GlobalMaxPooling1D()(output)
        output = Dense(self.n_classes, activation=None)(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    @overrides
    def save(self, fname=None):
        """
        Save the model parameters into <<fname>>_opt.json (or <<ser_file>>_opt.json)
        and model weights into <<fname>>.h5 (or <<ser_file>>.h5)
        Args:
            fname: file_path to save model. If not explicitly given seld.opt["ser_file"] will be used

        Returns:
            None
        """

        if not self.save_path:
            raise ConfigError("No `save_path` is provided for Keras model!")
        elif isinstance(self.save_path, Path) and not self.save_path.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        else:
            opt_path = "{}_opt.json".format(str(self.save_path.resolve()))
            weights_path = "{}.h5".format(str(self.save_path.resolve()))
            log.info("[saving model to {}]".format(opt_path))
            self.model.save_weights(weights_path)

        if type(self.opt["binary_mask"]) is list:
            pass
        else:
            self.opt["binary_mask"] = self.opt["binary_mask"].tolist()

        save_json(self.opt, opt_path)
        return True