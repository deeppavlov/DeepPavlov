import json
import os
import random
from typing import Dict, List, Tuple
from pathlib import Path
from logging import getLogger
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader

logger = getLogger(__name__)


@register('rured_reader')
class RuREDDatasetReader(DatasetReader):
    """ Class to read the datasets in RuRED format"""

    @overrides
    def read(self, data_path: str, rel2id: Dict = None) -> Dict[str, List[Tuple]]:
        """
        This class processes the RuRED relation extraction dataset
        (http://www.dialog-21.ru/media/5093/gordeevdiplusetal-031.pdf).
        Args:
            data_path: a path to a folder with dataset files.
            rel2id: a path to a file where information about relation to relation id corresponding is stored.
        Returns:
            RuRED output dictionary in the following format:
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

        data_path = Path(data_path).resolve()

        if not rel2id:
            self.rel2id = self.add_default_rel_dict()
        else:
            self.rel2id = rel2id
        self.stat = {}
        self.ner_stat = {}

        with open(os.path.join(data_path, "train.json"), encoding='utf-8') as file:
            train_data = json.load(file)

        with open(os.path.join(data_path, "dev.json"), encoding='utf-8') as file:
            dev_data = json.load(file)

        with open(os.path.join(data_path, "test.json"), encoding='utf-8') as file:
            test_data = json.load(file)

        train_data, self.stat["train"] = self.process_rured_file(train_data, num_neg_samples="twice")
        dev_data, self.stat["dev"] = self.process_rured_file(dev_data, num_neg_samples="equal")
        test_data, self.stat["test"] = self.process_rured_file(test_data, num_neg_samples="equal")

        data = {"train": train_data, "valid": dev_data, "test": test_data}

        return data

    def process_rured_file(self, data: List[Dict], num_neg_samples: str) -> Tuple[List, Dict]:
        """
        Processes a RuRED data and returns a DeepPavlov relevant output

        Args:
            data: List of data units
            num_neg_samples: how many negative samples will be included to positive ones
                Possible values:
                    - None: no negative samples will be generated
                        (relevant to the test set which has from neg samples only)
                    - equal: there will be one negative sample pro positive sample
                    - twice: there will be twice as many negative samples as positive ones
                    - all: take all negative samples from the dataset
        Returns:
            one list of processed documents
        """
        processed_samples = []
        neg_samples = []        # list of indices of negative samples
        pos_samples = 0         # counter of positive samples

        for sample in data:
            # record negative sample ids
            if sample["relation"] == "no_relation":
                neg_samples.append(len(processed_samples))
            else:
                pos_samples += 1

            if sample["subj_type"] in self.ner_stat:
                self.ner_stat[sample["subj_type"]] += 1
            else:
                self.ner_stat[sample["subj_type"]] = 1

            if sample["obj_type"] in self.ner_stat:
                self.ner_stat[sample["obj_type"]] += 1
            else:
                self.ner_stat[sample["obj_type"]] = 1

            processed_samples.append(
                (
                    [
                        sample["token"],
                        [[(sample["subj_start"], sample["subj_end"])], [(sample["obj_start"], sample["obj_end"])]],
                        [sample["subj_type"], sample["obj_type"]]
                    ],
                    self.label_to_one_hot(self.rel2id[sample["relation"]])
                )
            )

        # filter out some of negative sample if relevant
        if num_neg_samples == "equal":
            # include the same amount of negative samples as positive ones
            neg_to_eliminate = random.sample(neg_samples, (len(neg_samples) - pos_samples))
            processed_samples = [
                sample for sample_idx, sample in enumerate(processed_samples) if sample_idx not in neg_to_eliminate
            ]
        elif num_neg_samples == "twice":
            # include twice as much negative samples as positive ones
            neg_to_eliminate = random.sample(neg_samples, (len(neg_samples) - 2 * pos_samples))
            processed_samples = [
                sample for sample_idx, sample in enumerate(processed_samples) if sample_idx not in neg_to_eliminate
            ]
        elif num_neg_samples == "none":
            # eliminate all negative samples
            processed_samples = [
                sample for sample_idx, sample in enumerate(processed_samples) if sample_idx not in neg_samples
            ]
        else:
            raise ValueError("Unknown negative samples amount! Currently available are 'equal', 'twice' and 'none")

        # collect statistics
        stat = {}
        for sample in processed_samples:
            rel = [rel for rel, sample_log in enumerate(sample[1]) if sample_log == 1][0]
            if rel in stat:
                stat[rel] += 1
            else:
                stat[rel] = 1

        return processed_samples, stat

    def label_to_one_hot(self, label: int) -> List[int]:
        """ Turn labels to one hot encodings """
        relation = [0] * len(self.rel2id)
        relation[label] = 1
        return relation

    @staticmethod
    def add_default_rel_dict():
        """ Creates a default relation to relation if dictionary with RuRED relations """
        return dict(no_relation=0, MEMBER=1, WORKS_AS=2, WORKPLACE=3, OWNERSHIP=4, SUBORDINATE_OF=5, TAKES_PLACE_IN=6,
                    EVENT_TAKES_PART_IN=7, SELLS_TO=8, ALTERNATIVE_NAME=9, HEADQUARTERED_IN=10, PRODUCES=11,
                    ABBREVIATION=12, DATE_DEFUNCT_IN=13, SUBEVENT_OF=14, DATE_FOUNDED_IN=15, DATE_TAKES_PLACE_ON=16,
                    NUMBER_OF_EMPLOYEES_FIRED=17, ORIGINS_FROM=18, ACQUINTANCE_OF=19, PARENT_OF=20, ORGANIZES=21,
                    FOUNDED_BY=22, PLACE_RESIDES_IN=23, BORN_IN=24, AGE_IS=25, RELATIVE=26, NUMBER_OF_EMPLOYEES=27,
                    SIBLING=28, DATE_OF_BIRTH=29)
