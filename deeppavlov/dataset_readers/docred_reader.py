import itertools
import json
import os
import random
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple, Union
from overrides import overrides

import numpy as np
from sklearn.model_selection import train_test_split

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
            generate_additional_neg_samples: bool = False,
            num_neg_samples: int = None
    ) -> Dict[str, List[Tuple]]:
        """
        This class processes the DocRED relation extraction dataset (https://arxiv.org/abs/1906.06127v3).
        Args:
            data_path: a path to a folder with dataset files.
            rel2id_path: a path to a file where information about relation to relation id corresponding is stored.
            negative_label: a label which will be used as a negative one (by default in DocRED: "Na")
            generate_additional_neg_samples: boolean; whether to generate additional negative samples or not.
            num_neg_samples: a number of additional negative samples that will be generated for each positive sample.
        Returns:
            DocRED output dictionary in the following format:
            {"data_type":
                List[
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
                            ],
                        List(int(relation label))
                        )
                    ]
        """

        with open(str(expand_path(rel2id_path))) as file:
            self.rel2id = json.load(file)
        self.stat = {"POS_REL": 0, "NEG_REL": 0}  # collect statistics of positive and negative samples
        self.negative_label = negative_label
        self.if_add_neg_samples = generate_additional_neg_samples
        self.num_neg_samples = num_neg_samples

        if self.if_add_neg_samples and not self.num_neg_samples:
            raise ValueError("Please provide a number of negative samples to be generated!")

        data_path = Path(data_path).resolve()
        data = {"train": [], "valid": [], "test": []}

        # since in the original DocRED test data is given without labels, we will use a subset of train data instead
        data["train"], data["test"] = self.process_docred_file(
            os.path.join(data_path, "train_annotated.json"), split=0.1
        )
        data["valid"], _ = self.process_docred_file(os.path.join(data_path, "dev.json"))

        # todo: delete!
        # from joblib import dump
        # out = f"/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/docred/out_dataset_reader_without_neg/"
        # Path(out).mkdir(parents=True, exist_ok=True)
        # out = os.path.join(out, "all_data")
        # dump(data, out)

        # statistic info: POS_REL = 47133, NEG_REL = 1548307
        return data

    def process_docred_file(self, file_path: str, split: float = None) -> Tuple[List, Union[List, None]]:
        """
        Processes a DocRED file and returns a DeepPavlov relevant output

        Args:
            file_path: path to the file.
            split: in what proportion to split the dataset in two parts (relevant for the train data, which is splitted
                into train and dev sets; no splitting by default).

        Returns:
            if splitting: two lists of documents in DocRED output format (see documentation to the "read" function)
            if no splitting: one list of documents & None
        """
        processed_data_samples = []

        with open(file_path) as file:
            data = json.load(file)

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
                if len(data_unit["labels"]) == 0:
                    processed_data_samples += self.construct_neg_samples(ent_ids2ent_pos, ent_ids2ent_tag, doc)

                # if labels are provided, save samples as positive samples and generate negatives
                else:
                    labels = data_unit["labels"]
                    processed_data_samples += self.construct_pos_neg_samples(
                        labels, ent_ids2ent_pos, ent_ids2ent_tag, doc
                    )

        logger.info(f"Data: {os.path.split(file_path)[1]}. Positive samples: {self.stat['POS_REL']}. "
                    f"Negative samples: {self.stat['NEG_REL']}.")

        if split:
            return train_test_split(processed_data_samples, test_size=float(split), random_state=123)
        else:
            return processed_data_samples, None

    def construct_pos_neg_samples(self, labels: List, ent_id2ent: Dict, ent_ids2ent_tag: Dict, doc: List) -> List:
        """
        Transforms the relevant information into an entry of the DocRED reader output. The entities between which
        the relation is hold will serve as an annotation for positive samples, while all other entity pairs will be
        used to construct the negative samples.
        Args:
            labels: information about relation found in a document (whole labels list of the original DocRED)
            ent_id2ent: a dictionary {entity id: [entity mentions' positions]}
            ent_ids2ent_tag: a dictionary {entity id: entity NER tag}
            doc: list of all tokens of the document
        Returns:
            a tuple with list of all doc tokens, entity information (positions & NER tags) and relation.
        """
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
                label_one_hot = self.label_to_one_hot(rel_triples[(ent1, ent2)])
                data_samples.append(
                    (
                        (
                         doc,
                         [ent_id2ent[ent1], ent_id2ent[ent2], ent_ids2ent_tag[ent1], ent_ids2ent_tag[ent2]]
                        ),
                        label_one_hot
                    )
                )
                self.stat["POS_REL"] += 1

            # if there is no relation hold between entities, save them (and a corresponding sample) as negative one
            else:
                data_samples.append(
                    (
                        (
                         doc,
                         [ent_id2ent[ent1], ent_id2ent[ent2], ent_ids2ent_tag[ent1], ent_ids2ent_tag[ent2]]
                        ),
                        neg_label_one_hot
                    )
                )
                self.stat["NEG_REL"] += 1
        return data_samples

    def construct_neg_samples(
            self, ent_ids2ent: Dict, ent_ids2ent_tag: Dict, doc: List
    ) -> List[Tuple[List, List, List]]:
        """
        Turn the annotated documents but without any positive relation label to the negative samples in a format of
            the DocRED reader output.
        Args:
            ent_ids2ent: a dictionary {entity id: [entity mentions' positions]}
            ent_ids2ent_tag: a dictionary {entity id: entity NER tag}
            doc: list of all tokens of the document
        Returns:
            a tuple with list of all doc tokens, entity information (positions & NER tags) and relation (=neg_label).
        """
        neg_data_samples = []
        neg_label_one_hot = self.label_to_one_hot([self.rel2id[self.negative_label]])
        for ent1, ent2 in itertools.permutations(ent_ids2ent.keys(), 2):
            neg_data_samples.append(
                (
                    (
                     doc,
                     [ent_ids2ent[ent1], ent_ids2ent[ent2], ent_ids2ent_tag[ent1][0], ent_ids2ent_tag[ent2][0]]
                    ),
                    neg_label_one_hot
                )
            )
            self.stat["NEG_REL"] += 1
        return neg_data_samples

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
        "/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED",
        "/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/docred/meta/rel2id.json",
    )
