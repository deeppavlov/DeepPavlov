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

from io import StringIO
from typing import List, Tuple

from udapi.block.read.conllu import Conllu
from udapi.core.node import Node
from ufal_udpipe import Model as udModel, Pipeline

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


def descendents(node, desc_list):
    if len(node.children) > 0:
        for child in node.children:
            desc_list = descendents(child, desc_list)
    desc_list.append(node.form)

    return desc_list


@register('tree_parser')
class TreeParser(Component, Serializable):
    """
        This class parses the question using UDPipe to detect entity and relation
    """

    def __init__(self, load_path: str,
                 udpipe_filename: str, **kwargs) -> None:
        """
        
        Args:
            load_path: path to file with UDPipe model
            udpipe_filename: filename with UDPipe model
            **kwargs
        """
        super().__init__(save_path=None, load_path=load_path)
        self.udpipe_filename = udpipe_filename
        self.udpipe_load_path = self.load_path / self.udpipe_filename
        self.ud_model = udModel.load(str(self.udpipe_load_path))
        self.full_ud_model = Pipeline(self.ud_model, "vertical", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass

    def __call__(self, q_tokens: List[str],
                 *args, **kwargs) -> Tuple[str, str]:

        q_str = '\n'.join(q_tokens)
        s = self.full_ud_model.process(q_str)
        tree = Conllu(filehandle=StringIO(s)).read_tree()
        fnd, detected_entity, detected_rel = self.find_entity(tree, q_tokens)
        if fnd == False:
            fnd, detected_entity, detected_rel = self.find_entity_adj(tree)
        detected_entity = detected_entity.replace("первый ", '')
        return detected_entity, detected_rel

    def find_entity(self, tree: Node,
                    q_tokens: List[str]) -> Tuple[bool, str, str]:
        detected_entity = ""
        detected_rel = ""
        min_tree = 10
        leaf_node = None
        for node in tree.descendants:
            if len(node.children) < min_tree and node.upos in ["NOUN", "PROPN"]:
                leaf_node = node

        if leaf_node is not None:
            node = leaf_node
            desc_list = []
            entity_tokens = []
            while node.parent.upos in ["NOUN", "PROPN"] and node.parent.deprel != "root" \
                    and not node.parent.parent.form.startswith("Как"):
                node = node.parent
            detected_rel = node.parent.form
            desc_list.append(node.form)
            desc_list = descendents(node, desc_list)
            num_tok = 0
            for n, tok in enumerate(q_tokens):
                if tok in desc_list:
                    entity_tokens.append(tok)
                    num_tok = n
            if (num_tok + 1) < len(q_tokens):
                if q_tokens[(num_tok + 1)].isdigit():
                    entity_tokens.append(q_tokens[(num_tok + 1)])
            detected_entity = ' '.join(entity_tokens)
            return True, detected_entity, detected_rel

        return False, detected_entity, detected_rel

    def find_entity_adj(self, tree: Node) -> Tuple[bool, str, str]:
        detected_rel = ""
        detected_entity = ""
        for node in tree.descendants:
            if len(node.children) <= 1 and node.upos == "ADJ":
                detected_rel = node.parent.form
                detected_entity = node.form
                return True, detected_entity, detected_rel

        return False, detected_entity, detected_rel
