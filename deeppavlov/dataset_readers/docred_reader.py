import itertools
import json
import os
import random
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Any, Tuple
from overrides import overrides

import numpy as np
import spacy
from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader

# from deeppavlov import configs, build_model
# ner = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

# ATTENTION! To make it work, please run the following command: python3 -m spacy download en_core_web_sm
ner = spacy.load("en_core_web_sm")

logger = getLogger(__name__)

NEG_LABEL = "NO_REL"


@register('docred_reader')
class DocREDDatasetReader(DatasetReader):
    """ Class to read the datasets in DocRED format"""

    @overrides
    def read(
            self,
            data_path: str,
            generate_additional_neg_samples: bool = False,
            num_neg_samples: int = None
    ) -> Dict[str, List[Tuple]]:
        """
        This class processes the DocRED relation extraction dataset (https://arxiv.org/abs/1906.06127v3).
        Args:
            data_path: A path to a folder with dataset files.
            generate_additional_neg_samples: boolean; whether to generate additional negative samples or not.
            num_neg_samples:
        Returns:
            DocRED output dictionary.
        """
        # todo: add downloading from url

        self.stat = {"POS_REL": 0, "NEG_REL": 0}  # collect statistics of positive and negative samples
        self.if_add_neg_samples = generate_additional_neg_samples
        self.num_neg_samples = num_neg_samples

        if self.if_add_neg_samples and not self.num_neg_samples:
            raise ValueError("Please provide a number of negative samples to be generated!")

        data_path = Path(data_path)
        data = {"train": [], "dev": [], "test": []}

        data["train"], data["test"] = self.process_docred_file(
            os.path.join(data_path, "train_annotated.json"), split=0.1
        )
        data["dev"], _ = self.process_docred_file(os.path.join(data_path, "dev.json"))

        return data

    def process_docred_file(self, file_path: str, split: float = None) -> Tuple[List, Any]:
        """
        Processes a DocRED file and returns a DeepPavlov relevant output

        Args:
            file_path: path to the file.
            split: whether to split the dataset and in what proportion (relevant for the train data, which is splitted
                into train and dev sets).

        Returns:
            a list of documents represented as tuples of the following type:
                tuple[0] = tuple(list of sentences, list of entities, list of entity types),
                    where each entity = (start_idx, end_idx, text_entity)
                tuple[1] = list of relations matched in this document

        """
        processed_data_samples = []
        with open(file_path) as file:
            data = json.load(file)
            for data_unit in data:
                ent_ids2ent, ent_ids2ent_tag = {}, {}

                # get list of all tokens from the document
                doc = [token for sent in data_unit["sents"] for token in sent]

                # the sentence start indices are needed for entities' indices recalculation to the whole text
                sents_begins = list(np.cumsum([0] + [len(sent) + 1 for sent in data_unit["sents"]]))

                for ent_set_id, ent_set in enumerate(data_unit["vertexSet"]):
                    ent_ids2ent[ent_set_id] = [self.get_entity_info(ent, sents_begins) for ent in ent_set]
                    ent_ids2ent_tag[ent_set_id] = [ent["type"] for ent in ent_set]

                # if no labels are provided for the sample, handle is as a negative one
                if len(data_unit["labels"]) == 0:
                    processed_data_samples += self.construct_neg_samples(ent_ids2ent, ent_ids2ent_tag, doc)
                else:
                    for label_info in data_unit["labels"]:
                        processed_data_samples += self.construct_samples(ent_ids2ent, ent_ids2ent_tag, label_info, doc)

                # additionally generate negative samples for already included samples
                if self.if_add_neg_samples:
                    processed_data_samples += self.generate_additional_neg_samples(
                        doc, sum(ent_ids2ent.values(), []), self.num_neg_samples
                    )

        logger.info(f"Data: {os.path.split(file_path)[1]}. Positive samples: {self.stat['POS_REL']}. "
                    f"Negative samples: {self.stat['NEG_REL']}.")

        if split:
            return train_test_split(processed_data_samples, test_size=float(split), random_state=123)
        else:
            return processed_data_samples, None

    def construct_samples(self, ent_ids2ent: Dict, ent_ids2ent_tag: Dict, label_info: Dict, doc: List) -> List:
        """
        Args:
            ent_ids2ent:
            ent_ids2ent_tag:
            label_info:
            doc
        """
        data_samples = []
        entity1, entity2 = ent_ids2ent[label_info["h"]], ent_ids2ent[label_info["t"]]
        entity1_tag, entity2_tag = ent_ids2ent_tag[label_info["h"]], ent_ids2ent_tag[label_info["t"]]

        # construct output entry
        data_samples.append((doc, [entity1, entity2, entity1_tag[0], entity2_tag[0]], label_info['r']))

        self.stat["POS_REL"] += 1
        return data_samples

    @staticmethod
    def get_entity_info(ent: Dict, sents_begins: List) -> Tuple[int, int]:
        """
        The entity information as it is provided in DocRED is returned as:
            - the indices of the entity (recalculated regarding to the whole text)
            - the entity mention in the test

        Args:
            ent: a dictionary with entity information from DocRED
            sents_begins: a list of sentences' begin indices

        Returns:
            Tuple(entity_start_idx, entity_end_idx, entity_mention)

        """
        return (
            (ent["pos"][0] + sents_begins[ent["sent_id"]]),
            (ent["pos"][1] + sents_begins[ent["sent_id"]])
            # ent["name"]
        )

    def construct_neg_samples(self, ent_ids2ent: Dict, ent_ids2ent_tag: Dict, doc: List) -> List:
        neg_data_samples = []
        # construct output entry
        for ent1, ent2 in itertools.permutations(ent_ids2ent.keys(), 2):
            neg_data_samples.append(
                (doc, [ent_ids2ent[ent1], ent_ids2ent[ent2], ent_ids2ent_tag[ent1][0], ent_ids2ent_tag[ent2][0]],
                 NEG_LABEL)
            )
            self.stat["NEG_REL"] += 1
        return neg_data_samples

    def generate_additional_neg_samples(self, doc: List, pos_entities, num_neg_samples: int):
        """
        Generated negative samples, i.e. the same sentences that are used as positive ones, but labeled with
        "no_relation" label and with random tokens taken as entities.

        Args:
             doc: list of positive sentences
             pos_entities: list of entities that participate in any of the relations
             num_neg_samples:
        Returns:
             a list of documents in the same format as the positive ones.
        """
        neg_data_samples = []
        pos_ents_start, pos_ents_end = list(zip(*pos_entities))[0], list(zip(*pos_entities))[1]
        analysed_sentences = ner(" ".join(doc)).to_json()

        # select such pairs of tokens that were not part of any relation so far
        neg_spacy_entities = random.sample(
            [ent for ent in analysed_sentences["tokens"]
             if ent["start"] not in pos_ents_start or ent["end"] not in pos_ents_end],
            num_neg_samples
        )

        for n_ent_1, n_ent_2 in itertools.permutations(neg_spacy_entities, 2):
            if len(neg_data_samples) == num_neg_samples:
                break
            neg_entity_1 = (n_ent_1["start"], n_ent_1["end"])
            neg_entity_2 = (n_ent_2["start"], n_ent_2["end"])
            neg_entity_1_tag = n_ent_1["tag"]
            neg_entity_2_tag = n_ent_2["tag"]
            neg_data_samples.append(
                (doc, [[neg_entity_1], [neg_entity_2], neg_entity_1_tag, neg_entity_2_tag], NEG_LABEL)
            )
            self.stat["NEG_REL"] += 1

        return neg_data_samples


if __name__ == "__main__":
    DocREDDatasetReader().read(
        "/Users/asedova/Downloads/DocRED", generate_additional_neg_samples=True, num_neg_samples=5
    )

"""
ToDo: 
- change output format 
- instead of test data -> subset of train data 
- NER with DP
"""

"""
doc1: (ent1, ent2, rel1), (ent1, ent3, rel2)
(   
    [list of all tokens of a document1], 
    [
        [
            (ent1_pos1_start, ent1_pos1_end),
            (ent1_pos2_start, ent1_pos2_end)
        ],
        [
            (ent2_pos1_start, ent2_pos1_end)
        ],
        ner(ent1), 
        ner(ent2)
    ], 
    rel1
),
(
    [list of all tokens of a document1], 
    [ent1_pos_start, ent1_pos_end, ent3_pos_start, ent3_pos_end, ner(ent1), ner(ent3)], 
    rel2
)
"""
