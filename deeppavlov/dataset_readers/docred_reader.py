import json
import os
import random
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Any, Tuple
from overrides import overrides

import numpy as np
import spacy

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
    def read(self, data_path: str, generate_additional_neg_samples: bool = False) -> Dict[str, List[Tuple[Any, Any]]]:
        """
        This class processes the DocRED relation extraction dataset (https://arxiv.org/abs/1906.06127v3).
        Args:
            data_path: A path to a folder with dataset files.
            generate_additional_neg_samples: boolean; whether to generate additional negative samples or not.
        Returns:
            DocRED output dictionary.
        """
        # todo: add downloading from url

        data_path = Path(data_path)
        data = {"train": os.path.join(data_path, "train_annotated.json"),
                "dev": os.path.join(data_path, "dev.json"),
                "test": os.path.join(data_path, "test.json")}

        for data_type, data_file in data.items():
            if data_type == "test":  # there are no labels for test data
                data[data_type] = self.process_docred_file(
                    data_file, add_neg_samples=generate_additional_neg_samples, labels_provided=False
                )
            else:
                data[data_type] = self.process_docred_file(data_file, add_neg_samples=generate_additional_neg_samples)

        return data

    def process_docred_file(
            self, file_path: str, labels_provided: bool = True, add_neg_samples: bool = True
    ) -> List[Tuple[Any, Any]]:
        """
        Processes a DocRED file and returns a DeepPavlov relevant output

        Args:
            file_path: path to the file.
            labels_provided: whether the labels are included to the dataset (i.e., in train and dev sets).
            add_neg_samples: whether to generate additional negative samples.

        Returns:
            a list of documents represented as tuples of the following type:
                tuple[0] = tuple(list of sentences, list of entities, list of entity types),
                    where each entity = (start_idx, end_idx, text_entity)
                tuple[1] = list of relations matched in this document

        """
        data_samples = []
        stat = {"POS_REL": 0, "NEG_REL": 0}       # collect statistics of positive and negative samples
        with open(file_path) as file:
            data = json.load(file)
            for data_unit in data:
                entities, entitiy_tags = [], []
                sentences = [" ".join(sent) for sent in data_unit["sents"]]  # doc = list of sentences

                # the sentence start indices are needed for entities' indices recalculation to the whole text
                sents_begins = list(np.cumsum([0] + [len(sent) + 1 for sent in data_unit["sents"]]))

                for ent_set_id, ent_set in enumerate(data_unit["vertexSet"]):
                    entities += [self.get_entity_info(ent, sents_begins) for ent in ent_set]
                    entitiy_tags += [ent["type"] for ent in ent_set]

                if labels_provided:         # include labels if they are available
                    labels = [label["r"] for label in data_unit["labels"]]
                    if len(labels) == 0:        # if no labels are provided for the sample, handle is as a negative one
                        data_samples.append(((sentences, entities, entitiy_tags), [NEG_LABEL]))
                        stat["NEG_REL"] += 1
                    else:
                        data_samples.append(((sentences, entities, entitiy_tags), labels))
                        stat["POS_REL"] += 1
                else:
                    data_samples.append((sentences, entities, entitiy_tags))

                if add_neg_samples:         # if additional negative samples are to be generated and added to the data
                    data_samples += self.generate_negative_samples(sentences, entities, len(labels))

        logger.info(f"Data: {os.path.split(file_path)[1]}. Positive samples: {stat['POS_REL']}. "
                    f"Negative samples: {stat['NEG_REL']}.")

        return data_samples

    @staticmethod
    def get_entity_info(ent: Dict, sents_begins: List) -> Tuple[int, int, str]:
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
            (ent["pos"][1] + sents_begins[ent["sent_id"]]),
            ent["name"]
        )

    @staticmethod
    def generate_negative_samples(sentences: List, pos_entities: List, num_rel: int):
        """
        Generated negative samples, i.e. the same sentences that are used as positive ones, but labeled with
        "no_relation" label and with random tokens taken as entities.

        Args:
             sentences: list of positive sentences
             pos_entities: list of entities that participate in any of the relations
             num_rel: amount of positive relations detected in the set of sentences
        Returns:
             a list of documents in the same format as the positive ones.
        """
        pos_ents_start, pos_ents_end = list(zip(*pos_entities))[0], list(zip(*pos_entities))[1]
        analysed_sentences = ner(" ".join(sentences)).to_json()

        neg_spacy_entities = random.sample(
            [ent for ent in analysed_sentences["tokens"] if ent["start"] not in pos_ents_start or
                                                            ent["end"] not in pos_ents_end],
            len(pos_entities)
        )

        neg_entities = [(n_ent["start"], n_ent["end"]) for n_ent in neg_spacy_entities]
        neg_entitiy_tags = [n_ent["tag"] for n_ent in neg_spacy_entities]
        neg_samples = [((sentences, neg_entities, neg_entitiy_tags), [NEG_LABEL] * num_rel)]
        return neg_samples


if __name__ == "__main__":
    DocREDDatasetReader().read("/Users/asedova/Downloads/DocRED")



"""
doc1: (ent1, ent2, rel1), (ent1, ent3, rel2)
(   
    [list of all tokens of a document1], 
    [
        [
            (ent1_pos1_start, ent1_pos1_end),
            (ent1_pos2_start, ent1_pos2_end)
        ]
        [
            (ent2_pos1_start, ent2_pos1_end),
            (ent2_pos2_start, ent2_pos2_end)
        ]
        ner(ent1), 
        ner(ent2)
    ], 
    rel1
)
(
    [list of all tokens of a document1], 
    [ent1_pos_start, ent1_pos_end, ent3_pos_start, ent3_pos_end, ner(ent1), ner(ent3)], 
    rel2
)
"""