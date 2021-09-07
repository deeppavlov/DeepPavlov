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

import itertools
import json
import os
import random
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader

logger = getLogger(__name__)


@register('docred_reader')
class DocREDDatasetReader(DatasetReader):
    """ Class to read the datasets in DocRED format"""

    @overrides
    def read(
            self,
            data_path: str,
            rel2id_path: str,
            rel_info_path: str,
            negative_label: str = "Na",
            train_valid_test_proportion: int = None,
            valid_test_data_size: int = None,
            generate_additional_neg_samples: bool = False,
            num_neg_samples: int = None
    ) -> Dict[str, List[Tuple]]:
        """
        This class processes the DocRED relation extraction dataset (https://arxiv.org/abs/1906.06127v3).
        Args:
            data_path: a path to a folder with dataset files.
            rel2id_path: a path to a file where information about relation to relation id corresponding is stored.
            rel_info_path: a path to a file where information about relations and their real names is stored
            negative_label: a label which will be used as a negative one (by default in DocRED: "Na")
            train_valid_test_proportion: a proportion in which the data will be splitted into train, valid and test sets
            valid_test_data_size: absolute amount of dev & test sets
            generate_additional_neg_samples: boolean; whether to generate additional negative samples or not.
            num_neg_samples: a number of additional negative samples that will be generated for each positive sample.
        Returns:
            DocRED output dictionary in the following format:
            {"data_type":
                List[
                    Tuple(
                        List[
                            List[all tokens of the document],
                            List[
                                List[Tuple(start pos of mention 1 of ent 1, end pos of mention 1 of ent 1), ...],
                                List[Tuple(start position of entity 2, end position of entity 2), ...],
                                List[str(NER tag of entity 1), str(NER tag of entity 2)]
                            ],
                        List(int(one-hot encoded relation label))
                    )
                ]
            }
        """

        with open(str(expand_path(rel2id_path))) as file:
            self.rel2id = json.load(file)
        self.id2rel = {value: key for key, value in self.rel2id.items()}

        with open(str(expand_path(rel_info_path))) as file:
            self.relid2rel = json.load(file)
        self.rel2relid = {value: key for key, value in self.relid2rel.items()}

        self.negative_label = negative_label
        self.if_add_neg_samples = generate_additional_neg_samples
        self.num_neg_samples = num_neg_samples

        if self.if_add_neg_samples and not self.num_neg_samples:
            raise ValueError("Please provide a number of negative samples to be generated!")

        if train_valid_test_proportion and valid_test_data_size:
            raise ValueError(
                f"The train, valid and test splitting should be done either basing on their proportional values to each"
                f"other (train_valid_test_proportion parameter), or on the absolute size of valid and test data "
                f"(valid_test_data_size parameter). They can't be used simultaneously."
            )

        self.train_valid_test_proportion = train_valid_test_proportion
        self.valid_test_data_size = valid_test_data_size

        data_path = Path(data_path).resolve()

        with open(os.path.join(data_path, "train_annotated.json")) as file_ann:
            train_data = json.load(file_ann)

        with open(os.path.join(data_path, "dev.json")) as file:
            valid_data = json.load(file)

        # if you want to use test data from the original docred without labels (e.g. as negatives...),
        # uncomment lines below
        # with open(os.path.join(data_path, "test.json")) as file:
        #     test_data = json.load(file)
        #     test_processed = self.process_docred_file(test_data, neg_samples=None)

        # merge valid and train data and split them again into train, valid & test
        if self.train_valid_test_proportion:
            train_data, test_data, valid_data = self.split_by_relative(list(train_data + valid_data))
        elif self.valid_test_data_size:
            train_data, test_data, valid_data = self.split_by_absolute(list(train_data + valid_data))

        else:
            raise ValueError(
                f"The train, valid and test splitting should be done either basing on their proportional values to each"
                f"other (train_valid_test_proportion parameter), or on the absolute size of valid and test data "
                f"(valid_test_data_size parameter). One of them should be set to the not-None value."
            )

        logger.info("Train data processing...")
        train_data, train_stat = self.process_docred_file(train_data, neg_samples="twice")

        logger.info("Valid data processing...")
        valid_data, valid_stat = self.process_docred_file(valid_data, neg_samples="equal")

        logger.info("Test data processing...")
        test_data, test_stat = self.process_docred_file(test_data, neg_samples="equal")

        self.print_statistics(train_stat, valid_stat, test_stat)

        data = {"train": train_data, "valid": valid_data, "test": test_data}

        return data

    def split_by_absolute(self, all_labeled_data: List) -> Tuple[List, List, List]:
        """
        All annotated data from DocRED is splitted into train, valid and test sets in following proportions:
          len(valid_data) = len(test_data) = self.valid_test_data_size
          len(train_data) = len(all data) - 2 * self.valid_test_data_size
        Args:
            all_labeled_data: List of all annotated data samples
        Return:
            Lists of train, valid and test data
        """
        if (int(self.valid_test_data_size) * 3) > len(all_labeled_data):
            raise ValueError(
                f"The dataset size {len(all_labeled_data)} is too small for taking {self.valid_test_data_size} samples"
                f"for valid and test. Reduce the size of valid and test set."
            )

        random.shuffle(all_labeled_data)
        valid_data = all_labeled_data[:int(self.valid_test_data_size)]
        test_data = all_labeled_data[int(self.valid_test_data_size) + 1: 2 * int(self.valid_test_data_size)]
        train_data = all_labeled_data[2 * int(self.valid_test_data_size) + 1:]
        return train_data, valid_data, test_data

    def split_by_relative(self, all_labeled_data: List) -> Tuple[List, List, List]:
        """
        All annotated data from DocRED is splitted into train, valid and test sets in following proportions:
          len(train_data) = train_valid_test_proportion * len(valid_data) = train_valid_test_proportion * len(test_data)
        """
        random.shuffle(all_labeled_data)
        one_prop = int(len(all_labeled_data)/int(self.train_valid_test_proportion))

        valid_data = all_labeled_data[:one_prop]
        test_data = all_labeled_data[one_prop + 1: 2 * one_prop]
        train_data = all_labeled_data[2 * one_prop + 1:]
        return train_data, valid_data, test_data

    def process_docred_file(self, data: List[Dict], neg_samples: str = None) -> Tuple[List, Dict]:
        """
        Processes a DocRED data and returns a DeepPavlov relevant output

        Args:
            data: List of data units
            neg_samples: how many negative samples are to be generated
                Possible values:
                    - None: no negative samples will be generated
                        (relevant to the test set which has from neg samples only)
                    - equal: there will be one negative sample pro positive sample
                    - twice: there will be twice as many negative samples as positive ones
                    - thrice: there will be thrice as many negative samples as positive ones
        Returns:
            one list of processed documents
        """
        stat_rel_name = {rel_name: 0 for _, rel_name in self.relid2rel.items()}
        self.stat = {"POS_REL": 0, "NEG_REL": 0}  # collect statistics of positive and negative samples
        processed_data_samples = []

        for data_unit in data:
            ent_ids2ent_pos, ent_ids2ent_text, ent_ids2ent_tag = {}, {}, {}

            # get list of all tokens from the document
            doc = [token for sent in data_unit["sents"] for token in sent]

            # the sentence start indices are needed for entities' indices recalculation to the whole text
            sents_begins = list(np.cumsum([0] + [len(sent) for sent in data_unit["sents"]]))

            for ent_set_id, ent_set in enumerate(data_unit["vertexSet"]):
                ent_ids2ent_pos[ent_set_id], ent_ids2ent_text[ent_set_id], ent_ids2ent_tag[ent_set_id] = [], [], []
                for ent in ent_set:
                    # the list of tuples with each entity's new indices (recalculated regarding to the whole doc)
                    ent_ids2ent_pos[ent_set_id].append(
                        ((ent["pos"][0] + sents_begins[ent["sent_id"]]),
                         (ent["pos"][1] + sents_begins[ent["sent_id"]]))
                    )
                    # also save entity id to entity as exact text mentions correspondence
                    ent_ids2ent_text[ent_set_id].append(ent["name"])
                # get the sample NER tag (logically, the same for all entity mentions)
                ent_ids2ent_tag[ent_set_id] = ent_set[0]["type"]
                ent_ids2ent_text[ent_set_id] = list(set(ent_ids2ent_text[ent_set_id]))

            # if no labels are provided for the data, handle all samples as negative ones
            if "labels" not in data_unit:
                processed_data_samples += self.construct_neg_samples(ent_ids2ent_pos, ent_ids2ent_tag, doc)

            # if labels are provided, save samples as positive samples and generate negatives
            else:
                labels = data_unit["labels"]
                curr_processed_data_samples, stat_rel_name = self.construct_pos_neg_samples(
                    labels, ent_ids2ent_pos, ent_ids2ent_tag, doc, stat_rel_name, neg_samples=neg_samples,
                )
                processed_data_samples += curr_processed_data_samples

        logger.info(f"Pos samples: {self.stat['POS_REL']}  Neg samples: {self.stat['NEG_REL']}.")
        self.stat.pop("POS_REL")
        self.stat.pop("NEG_REL")

        return processed_data_samples, stat_rel_name

    def construct_pos_neg_samples(
            self, labels: List, ent_id2ent: Dict, ent_id2ent_tag: Dict, doc: List, stat_rel: Dict, neg_samples: str,
    ) -> Tuple[List, Dict]:
        """
        Transforms the relevant information into an entry of the DocRED reader output. The entities between which
        the relation is hold will serve as an annotation for positive samples, while all other entity pairs will be
        used to construct the negative samples.

        Args:
            labels: information about relation found in a document (whole labels list of the original DocRED)
            ent_id2ent: a dictionary {entity id: [entity mentions' positions]}
            stat_rel: a dictionary with relation statistics (will be updated)
            neg_samples: amount of negative samples that are to be generated
            ent_id2ent_tag: a dictionary {entity id: entity NER tag}
            doc: list of all tokens of the document
        Returns:
            a tuple with list of all doc tokens, entity information (positions & NER tags) and relation.
        """

        num_pos_samples, num_neg_samples = 0, 0

        data_samples = []
        rel_triples = {}
        for label_info in labels:
            entity1_id, entity2_id = label_info["h"], label_info["t"]
            if (entity1_id, entity2_id) in rel_triples:
                rel_triples[(entity1_id, entity2_id)].append(self.rel2id[label_info['r']])
            else:
                rel_triples[(entity1_id, entity2_id)] = [self.rel2id[label_info['r']]]

        # the one hot encoding of the negative label
        neg_label_one_hot = self.label_to_one_hot([self.rel2id[self.negative_label]])

        # iterate over all entities
        for (ent1, ent2) in itertools.permutations(ent_id2ent, 2):

            # if there is a relation hold between entities, save them (and a corresponding sample) as positive one
            if (ent1, ent2) in rel_triples:
                num_pos_samples += 1
                labels = rel_triples[(ent1, ent2)]
                label_one_hot = self.label_to_one_hot(labels)
                data_samples.append(
                    self.generate_data_sample(doc, ent1, ent2, label_one_hot, ent_id2ent, ent_id2ent_tag)
                )
                self.stat["POS_REL"] += 1

                for label in labels:
                    rel_name = self.relid2rel[self.id2rel[label]]
                    stat_rel[rel_name] += 1

            else:
                if not neg_samples:         # if no negative samples should be generated, skip
                    continue

                # if there is no relation hold between entities, save them (and a corresponding sample) as negative one
                if neg_samples == "equal" and num_neg_samples < num_pos_samples:
                    num_neg_samples += 1
                    data_samples.append(
                        self.generate_data_sample(doc, ent1, ent2, neg_label_one_hot, ent_id2ent, ent_id2ent_tag)
                    )
                    self.stat["NEG_REL"] += 1

                elif neg_samples == "twice" and num_neg_samples < 2 * num_pos_samples:
                    num_neg_samples += 1
                    data_samples.append(
                        self.generate_data_sample(doc, ent1, ent2, neg_label_one_hot, ent_id2ent, ent_id2ent_tag)
                    )
                    self.stat["NEG_REL"] += 1

                elif neg_samples == "thrice" and num_neg_samples < 3 * num_pos_samples:
                    num_neg_samples += 1
                    data_samples.append(
                        self.generate_data_sample(doc, ent1, ent2, neg_label_one_hot, ent_id2ent, ent_id2ent_tag)
                    )
                    self.stat["NEG_REL"] += 1

        return data_samples, stat_rel

    def construct_neg_samples(
            self, ent_id2ent: Dict, ent_id2ent_tag: Dict, doc: List
    ) -> List[Tuple[Tuple[List, List], List]]:
        """
        Turn the annotated documents but without any positive relation label to the negative samples in a format of
            the DocRED reader output.

        Args:
            ent_id2ent: a dictionary {entity id: [entity mentions' positions]}
            ent_id2ent_tag: a dictionary {entity id: entity NER tag}
            doc: list of all tokens of the document
        Returns:
            a tuple with list of all doc tokens, entity information (positions & NER tags) and relation (=neg_label).
        """
        neg_data_samples = []
        neg_label_one_hot = self.label_to_one_hot([self.rel2id[self.negative_label]])
        for ent1, ent2 in itertools.permutations(ent_id2ent.keys(), 2):
            neg_data_samples.append(
                self.generate_data_sample(doc, ent1, ent2, neg_label_one_hot, ent_id2ent, ent_id2ent_tag)
            )

            self.stat["NEG_REL"] += 1
        return neg_data_samples

    @staticmethod
    def generate_data_sample(
            doc: List, ent1: int, ent2: int, label: List, ent_id2ent: Dict, ent_id2ent_tag: Dict
    ) -> Tuple[List[Union[List, List]], List]:
        """ Creates an entry of processed docred corpus """
        return (
                    [
                        doc,
                        [ent_id2ent[ent1], ent_id2ent[ent2]],
                        [ent_id2ent_tag[ent1], ent_id2ent_tag[ent2]]
                    ],
                    label
                )

    def generate_additional_neg_samples(self, doc: List, forbidden_entities: List, num_neg_samples: int):
        """
        <CURRENTLY NOT USED>
        Generated negative samples, i.e. the same document that is used for positive samples, but labeled with
        "no_relation" label and with entities, that are not connected with any relation, marked as such.

        Args:
             doc: list of positive sentences
             forbidden_entities: list of entities that participate in any of the relations (and, therefore, cannot be
                chosen for negative sample)
             num_neg_samples: number of negative samples that are to be generated out of this document
        Returns:
             a tuple with list of all doc tokens, entity information (positions & NER tags) and relation (=neg_label).
        """
        # ATTENTION! To make it work, please run the following command: python3 -m deeppavlov install ner_ontonotes_bert

        from deeppavlov import build_model, configs
        ner = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
        neg_data_samples = []
        analysed_sentences = ner([" ".join(doc)])  # returns [[[tokens]], [[ner tags]]]

        # select ids of tokens that were not part of any relation so far
        neg_entities_idx = random.sample(
            [ent_idx for ent_idx in range(len(analysed_sentences[0][0]))
             if analysed_sentences[0][0][ent_idx] not in forbidden_entities],
            num_neg_samples * 2
        )

        # the one hot encoding of the negative label
        neg_label_one_hot = self.label_to_one_hot([self.rel2id[self.negative_label]])

        for n_ent_1_idx, n_ent_2_idx in itertools.permutations(neg_entities_idx, 2):
            # if already sufficient number of negative samples have been generated
            if len(neg_data_samples) == num_neg_samples:
                break
            neg_entity_1 = analysed_sentences[0][0][n_ent_1_idx]
            neg_entity_2 = analysed_sentences[0][0][n_ent_2_idx]
            neg_entity_1_tag = analysed_sentences[1][0][n_ent_1_idx]
            neg_entity_2_tag = analysed_sentences[1][0][n_ent_2_idx]
            neg_data_samples.append(
                (doc, [[neg_entity_1], [neg_entity_2], neg_entity_1_tag, neg_entity_2_tag], neg_label_one_hot)
            )
            self.stat["NEG_REL"] += 1

        return neg_data_samples

    def label_to_one_hot(self, labels: List[int]) -> List:
        """ Turn labels to one hot encodings """
        relation = [0] * len(self.rel2id)
        for label in labels:
            relation[label] = 1
        return relation

    def print_statistics(self, train_stat: Dict, valid_stat: Dict, test_stat: Dict) -> None:
        """ Print out the relation statistics as a markdown table """
        df = pd.DataFrame([self.rel2relid, train_stat, valid_stat, test_stat]).T
        df.columns = ['d{}'.format(i) for i, col in enumerate(df, 1)]
        logger.info("\n")
        logger.info(df)
