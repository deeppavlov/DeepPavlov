import itertools
import json
import os
import random
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple
from overrides import overrides

import numpy as np
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
            negative_label: str = "Na",
            train_dev_test_proportion: int = 7,
            generate_additional_neg_samples: bool = False,
            num_neg_samples: int = None
    ) -> Dict[str, List[Tuple]]:
        """
        This class processes the DocRED relation extraction dataset (https://arxiv.org/abs/1906.06127v3).
        Args:
            data_path: a path to a folder with dataset files.
            rel2id_path: a path to a file where information about relation to relation id corresponding is stored.
            negative_label: a label which will be used as a negative one (by default in DocRED: "Na")
            train_dev_test_proportion: a proportion in which the data will be splitted into train, dev and test sets
            generate_additional_neg_samples: boolean; whether to generate additional negative samples or not.
            num_neg_samples: a number of additional negative samples that will be generated for each positive sample.
        Returns:
            DocRED output dictionary in the following format:
            {"data_type":
                List[
                    Tuple(
                        Tuple(
                            List[all tokens of the document],
                            List[
                                List[
                                    # Tuples with information about text mentions of entity 1.
                                    # E.g., if entity 1 was mentioned two times in the document:
                                    Tuple(start position of mention 1 of entity 1, end position of mention 1 of entity 1),
                                    Tuple(start position of mention 2 of entity 1, end position of mention 2 of entity 1)
                                    ],
                                List[
                                    # Tuples with information about text mentions of entity 2
                                    # E.g., if entity 2 was mentioned once in the document:
                                    Tuple(start position of entity 2, end position of entity 2),
                                    ],
                                str(NER tag of entity 1),
                                str(NER tag of entity 2)
                                ]
                            ),
                        List(int(one-hot encoded relation label))
                        )
                    ]
        """

        with open(str(expand_path(rel2id_path))) as file:
            self.rel2id = json.load(file)
        self.negative_label = negative_label
        self.if_add_neg_samples = generate_additional_neg_samples
        self.num_neg_samples = num_neg_samples

        if self.if_add_neg_samples and not self.num_neg_samples:
            raise ValueError("Please provide a number of negative samples to be generated!")

        data_path = Path(data_path).resolve()

        with open(os.path.join(data_path, "train_annotated.json")) as file_ann:
            train_data = json.load(file_ann)
        with open(os.path.join(data_path, "train_distant.json"), encoding="UTF-8") as file_ds:
            train_data += json.load(file_ds)

        with open(os.path.join(data_path, "dev.json")) as file:
            dev_data = json.load(file)

        with open(os.path.join(data_path, "test.json")) as file:
            test_data = json.load(file)
            # process test data without labels (maybe use later as negatives...)
            test_processed = self.process_docred_file(test_data, neg_samples=None)

        # merge dev and train data and split them again so that:
        # len(train_data) = train_dev_test_proportion * len(dev_data) = train_dev_test_proportion * len(test_data)
        all_labeled_data = train_data + dev_data
        random.shuffle(all_labeled_data)
        one_prop = int(len(all_labeled_data)/train_dev_test_proportion)

        dev_data = all_labeled_data[:one_prop]
        test_data = all_labeled_data[one_prop + 1: 2 * one_prop]
        train_data = all_labeled_data[2 * one_prop + 1:]

        data = {
            "train": self.process_docred_file(train_data, neg_samples="thrice", data_type="train"),
            "valid": self.process_docred_file(dev_data, neg_samples="equal", data_type="valid"),
            "test": self.process_docred_file(test_data, neg_samples="equal", data_type="test")
        }

        # todo: delete!
        # from joblib import dump
        # out = f"/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/docred/out_dataset_reader_without_neg/"
        # Path(out).mkdir(parents=True, exist_ok=True)
        # out = os.path.join(out, "all_data")
        # dump(data, out)

        # statistic info: POS_REL = 47133, NEG_REL = 1548307
        return data

    def process_docred_file(self, data: List[Dict], neg_samples: str = None, data_type: str = None) -> List:
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
                processed_data_samples += self.construct_pos_neg_samples(
                    labels, ent_ids2ent_pos, ent_ids2ent_tag, doc, neg_samples=neg_samples
                )

        if data_type:
            logger.info(f"Data: {data_type}  Pos samples: {self.stat['POS_REL']}  Neg samples: {self.stat['NEG_REL']}.")

        self.stat.pop("POS_REL")
        self.stat.pop("NEG_REL")

        return processed_data_samples

    def construct_pos_neg_samples(
            self, labels: List, ent_id2ent: Dict, ent_id2ent_tag: Dict, doc: List, neg_samples: str
    ) -> List:
        """
        Transforms the relevant information into an entry of the DocRED reader output. The entities between which
        the relation is hold will serve as an annotation for positive samples, while all other entity pairs will be
        used to construct the negative samples.
        Args:
            labels: information about relation found in a document (whole labels list of the original DocRED)
            ent_id2ent: a dictionary {entity id: [entity mentions' positions]}
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
                label_one_hot = self.label_to_one_hot(rel_triples[(ent1, ent2)])
                data_samples.append(
                    self.generate_data_sample(doc, ent1, ent2, label_one_hot, ent_id2ent, ent_id2ent_tag)
                )
                self.stat["POS_REL"] += 1

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

        return data_samples

    def construct_neg_samples(
            self, ent_id2ent: Dict, ent_id2ent_tag: Dict, doc: List
    ) -> List[Tuple[List, List, List]]:
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

    def generate_data_sample(self, doc, ent1, ent2, label, ent_id2ent, ent_id2ent_tag):
        return (
            (
                doc,
                [ent_id2ent[ent1], ent_id2ent[ent2], ent_id2ent_tag[ent1], ent_id2ent_tag[ent2]]
            ),
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


# todo: wil be deleted
if __name__ == "__main__":
    DocREDDatasetReader().read(
        "/Users/asedova/PycharmProjects/05_deeppavlov_fork/docred",
        "/Users/asedova/PycharmProjects/05_deeppavlov_fork/docred/meta/rel2id.json",
    )
