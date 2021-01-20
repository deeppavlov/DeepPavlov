import bz2
import json
import multiprocessing as mp
import os
from logging import getLogger
from pathlib import Path
from deeppavlov.core.common.file import save_pickle

log = getLogger(__name__)

class WikidataParser:
    """
    This class parses json file with Wikidata
    and makes wiki_dict as a result
    wiki_dict is a dict with keys as entity_ids and values as entity_info
    an example of wiki_dict element:
        "Q649": {"name": "Москва", "aliases": ["Первопрестольная", "Порт пяти морей", "Москва (город)", "Москва, Россия"],
                 "descr": "cтолица и крупнейший город России", "wikipedia_title": "https://ru.wikipedia.org/wiki/Москва",
                 "triplets": [["P31", "Q4442912", "Q183342", "Q1549591"]], "number_of_relations": 558}
    """

    def __init__(self, wikidata_filename: str,
                 chunk_num_lines: int = 60000,
                 save_path: str = "~/.deeppavlov/downloads/wikidata_parse",
                 num_processors=None):

        self.wikidata_filename = wikidata_filename
        self.chunk_num_lines = chunk_num_lines
        self.save_path = Path(save_path).expanduser().resolve()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.manager = mp.Manager()
        if num_processors is None:
            self.num_processors = mp.cpu_count()
        else:
            self.num_processors = num_processors
        log.debug(f"number of processors {self.num_processors}")
        self.wiki_dict = self.manager.dict()
        self.bz_file = bz2.BZ2File(self.wikidata_filename)

    def process_sample(self, entity_dict):
        """
        Method which takes as input line from Wikidata file, parsed with json.loads (entity_dict)
        and extracts useful elements from it (entity_id and entity_info)
        """
        entity_info = {}
        entity_id = ""
        entity_in_russian = False
        try:
            if "id" in entity_dict:
                entity_id = entity_dict["id"]
                if "labels" in entity_dict:
                    if "ru" in entity_dict["labels"]:
                        name = entity_dict["labels"]["ru"]["value"]
                        entity_info["name"] = name
                        entity_in_russian = True
                if "aliases" in entity_dict:
                    if "ru" in entity_dict["aliases"]:
                        aliases = [alias["value"] for alias in entity_dict["aliases"]["ru"]]
                        entity_info["aliases"] = aliases
                if "descriptions" in entity_dict:
                    if "ru" in entity_dict["descriptions"]:
                        descr = entity_dict["descriptions"]["ru"]["value"]
                        entity_info["descr"] = descr

                wikipedia_title = entity_dict.get("sitelinks", {}).get("ruwiki", {}).get("title", "")
                entity_info["wikipedia_title"] = wikipedia_title

                if "claims" in entity_dict and entity_in_russian:
                    triplets = []
                    for relation in entity_dict["claims"]:
                        if relation in ["P31", "P279", "P106"]:
                            objects_list = []
                            objects = entity_dict["claims"][relation]
                            for obj in objects:
                                if "id" in obj["mainsnak"]["datavalue"]["value"]:
                                    objects_list.append(obj["mainsnak"]["datavalue"]["value"]["id"])
                            if objects_list:
                                triplets.append([relation] + objects_list)
                    entity_info["triplets"] = triplets

                    number_of_relations = len(entity_dict["claims"])
                    entity_info["number_of_relations"] = number_of_relations

        except:
            pass

        return entity_id, entity_info

    def run(self, num_proc, common_list):
        """
        Method for parallel processing of lines from Wikidata file
        """
        length = len(common_list)
        chunk_size = length // self.num_processors + 1
        for i in range(chunk_size):
            num_sample = self.num_processors * i + num_proc
            if num_sample < length:
                line = common_list[num_sample]
                line = line[:-2]
                entity = json.loads(line)
                entity_id, entity_info = self.process_sample(entity)
                if entity_id:
                    self.wiki_dict[entity_id] = entity_info

    def parse(self, continue_parsing: bool = False):
        """
        Method for parsing of Wikidata file
        """
        line = self.bz_file.readline()

        num_iterations = 0
        if continue_parsing:
            files = os.listdir(self.save_path)
            num_iterations = len(files)

            for _ in range(num_iterations * self.chunk_num_lines * self.num_processors):
                line = self.bz_file.readline()

        while True:
            log.debug(f"iteration number {num_iterations}")
            self.wiki_dict = self.manager.dict()
            common_list = []

            count = 0
            while line:
                line = self.bz_file.readline()
                common_list.append(line)
                count += 1
                if count == self.chunk_num_lines * self.num_processors:
                    break

            if not common_list:
                break

            workers = []
            for ii in range(self.num_processors):
                worker = mp.Process(target=self.run, args=(ii, common_list))
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()

            self.wiki_dict = dict(self.wiki_dict)

            save_pickle(self.wiki_dict, self.save_path / f"{num_iterations}.pickle")

            num_iterations += 1
