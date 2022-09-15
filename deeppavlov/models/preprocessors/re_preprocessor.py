# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
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
from logging import getLogger
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
from transformers import BertTokenizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('re_preprocessor')
class REPreprocessor(Component):
    def __init__(
            self,
            vocab_file: str,
            special_token: str = '<ENT>',
            ner_tags=None,
            max_seq_length: int = 512,
            do_lower_case: bool = False,
            default_tag: str = None,
            **kwargs
    ):
        """
        Args:
            vocab_file: path to vocabulary / name of vocabulary for tokenizer initialization
            special_token: an additional token that will be used for marking the entities in the document
            do_lower_case: set True if lowercasing is needed
            default_tag: used for test purposes to create a valid input
        Return:
            list of feature batches with input_ids, attention_mask, entity_pos, ner_tags
        """

        self.special_token = special_token
        self.special_tokens_dict = {'additional_special_tokens': [self.special_token]}
        self.default_tag = default_tag

        if ner_tags is None:
            ner_tags = ['ORG', 'TIME', 'MISC', 'LOC', 'PER', 'NUM']
        self.ner2id = {tag: tag_id for tag_id, tag in enumerate(ner_tags)}
        self.max_seq_length = max_seq_length

        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(
            self, tokens: Union[Tuple, List[List[str]]], entity_pos: Union[Tuple, List[List[Tuple]]],
            entity_tags: Union[Tuple, List[List[str]]],
    ) -> Tuple[List, List, List, List, List]:
        """
        Tokenize and create masks; recalculate the entity positions regarding the document boarders.
        Args:
            tokens: List of tokens of each document: List[List[tokens in doc]]
            entity_pos: start and end positions of the entities' mentions
            entity_tags: NER tag of the entities
        Return:
            input_ids: List[List[int]],
            attention_mask: List[List[int]],
            entity_poss: List[
                            List[
                                List[(entity1_mention1_start_id, entity1_mention1_end_id), ...],
                                List[(entity2_mention1_start_id, entity2_mention1_end_id), ...]
                            ]
                        ]
            entity_tags: List[List[int]]
            nf_samples: List[int] - contains the information about whether the corresponding sample is real sample or
                fake (for testing): 0 means the sample is real, 1 - it is fake.
        """

        _ = self.tokenizer.add_special_tokens(self.special_tokens_dict)

        input_ids, attention_mask, upd_entity_pos, upd_entity_tags, nf_samples = [], [], [], [], []

        # this workaround is for proper testing: for an unknown reason during test in test_quick_start.py
        # each input list is transformed into a tuple, e.g., tokens -> tuple(tokens, ).
        # todo: refactoring
        if type(tokens) == tuple and type(entity_pos) == tuple and type(entity_tags) == tuple:
            tokens = tokens[0]
            entity_pos = entity_pos[0]
            entity_tags = entity_tags[0]

        for n_sample, (doc, ent_pos, ent_tags) in enumerate(zip(tokens, entity_pos, entity_tags)):

            # valid scenario
            if isinstance(ent_pos, list) and len(ent_pos) == 2:
                count = 0
                doc_wordpiece_tokens = []

                entity1_pos_start = list(zip(*ent_pos[0]))[0]  # first entity mentions' start positions
                entity1_pos_end = list(zip(*ent_pos[0]))[1]  # first entity mentions' end positions
                entity2_pos_start = list(zip(*ent_pos[1]))[0]  # second entity mentions' start positions
                entity2_pos_end = list(zip(*ent_pos[1]))[1]  # second entity mentions' end positions

                upd_entity1_pos_start, upd_entity2_pos_start, upd_entity1_pos_end, upd_entity2_pos_end = [], [], [], []
                for n, token in enumerate(doc):
                    if n in entity1_pos_start:
                        doc_wordpiece_tokens.append(self.special_token)
                        upd_entity1_pos_start.append(count)
                        count += 1

                    if n in entity1_pos_end:
                        doc_wordpiece_tokens.append(self.special_token)
                        count += 1
                        upd_entity1_pos_end.append(count)

                    if n in entity2_pos_start:
                        doc_wordpiece_tokens.append(self.special_token)
                        upd_entity2_pos_start.append(count)
                        count += 1

                    if n in entity2_pos_end:
                        doc_wordpiece_tokens.append(self.special_token)
                        count += 1
                        upd_entity2_pos_end.append(count)

                    word_tokens = self.tokenizer.tokenize(token)
                    doc_wordpiece_tokens += word_tokens
                    count += len(word_tokens)

                # special case when the entity is the last in the doc
                if len(doc) in entity1_pos_end:
                    doc_wordpiece_tokens.append(self.special_token)
                    count += 1
                    upd_entity1_pos_end.append(count)
                if len(doc) in entity2_pos_end:
                    doc_wordpiece_tokens.append(self.special_token)
                    count += 1
                    upd_entity2_pos_end.append(count)
                    word_tokens = self.tokenizer.tokenize(token)
                    doc_wordpiece_tokens += word_tokens
                    count += len(word_tokens)

                upd_entity_1_pos = list(zip(upd_entity1_pos_start, upd_entity1_pos_end))
                upd_entity_2_pos = list(zip(upd_entity2_pos_start, upd_entity2_pos_end))

                # text entities for self check
                upd_entity1_text = [doc_wordpiece_tokens[ent_m[0]:ent_m[1]] for ent_m in upd_entity_1_pos]
                upd_entity2_text = [doc_wordpiece_tokens[ent_m[0]:ent_m[1]] for ent_m in upd_entity_2_pos]

                enc_entity_tags = self.encode_ner_tag(ent_tags)

                encoding = self.tokenizer.encode_plus(
                    doc_wordpiece_tokens[:self.max_seq_length],     # truncate tokens
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    pad_to_max_length=True,
                    return_attention_mask=True
                )
                upd_entity_pos.append([upd_entity_1_pos, upd_entity_2_pos])
                nf_samples.append(0)

            # api test scenario
            else:
                # for api test: dump values of entity tags and entity pos
                encoding = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    pad_to_max_length=True,
                    return_attention_mask=True
                )
                upd_entity_pos.append([[(0, 1)], [(0, 1)]])
                enc_entity_tags = self.encode_ner_tag([self.default_tag] * 2)
                nf_samples.append(1)

            input_ids.append(encoding['input_ids'])
            attention_mask.append(encoding['attention_mask'])
            upd_entity_tags.append(enc_entity_tags)

        return input_ids, attention_mask, upd_entity_pos, upd_entity_tags, nf_samples

    def encode_ner_tag(self, ner_tags: List) -> List:
        """ Encode NER tags with one hot encodings """
        enc_ner_tags = []
        for ner_tag in ner_tags:
            ner_tag_one_hot = [0] * len(self.ner2id)
            ner_tag_one_hot[self.ner2id[ner_tag]] = 1
            enc_ner_tags.append(ner_tag_one_hot)
        return enc_ner_tags


@register('re_postprocessor')
class REPostprocessor:

    def __init__(self, rel2id_path: str, rel2label_path: str, **kwargs):
        self.rel2id_path = rel2id_path
        self.rel2label_path = rel2label_path
        self.rel2id = read_json(str(expand_path(self.rel2id_path)))
        self.id2rel = {rel_id: rel for rel, rel_id in self.rel2id.items()}
        self.rel2label = read_json(str(expand_path(self.rel2label_path)))

    def __call__(self, model_output: List, nf_samples: List) -> Tuple[List[str], List[str]]:
        """
        The model output is transformed to the relation id and relation name
        Args:
            model_output: List of probability vectors
            nf_samples: contains the information about true and fake samples (0 - true sample and should be included
                to the output, 1 - fake sample)
        Return:
            wikidata_relation_id: List of wiki ids of found relations
            relation_name: List of names of found relations
        """

        wikidata_relation_id, relation_name = [], []

        for predictions, nf_sample in zip(model_output, nf_samples):
            if nf_sample:
                wikidata_relation_id.append("-")
                relation_name.append("-")
            else:
                rel_indices = np.nonzero(predictions)[0]

                for index in rel_indices:
                    if index == 0:
                        wikidata_relation_id.append("-")
                        relation_name.append("no relation")
                        continue

                    rel_p = self.id2rel[index]
                    wikidata_relation_id.append(rel_p)

                    if rel_p in self.rel2label:
                        relation_name.append(self.rel2label[rel_p])
                    else:
                        relation_name.append("-")
        return wikidata_relation_id, relation_name
