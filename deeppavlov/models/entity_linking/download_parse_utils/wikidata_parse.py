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
                 total_lines_num: int = 86000000,
                 save_path: str = "~/.deeppavlov/downloads/wikidata_parse",
                 log_path: str = "~/.deeppavlov/downloads/wikidata_parse_logs",
                 log_parse_errors: str = "log_parse_errors.txt",
                 log_parse_progress: str = "log_parse_progress.txt",
                 num_processors=None):

        self.wikidata_filename = wikidata_filename
        self.chunk_num_lines = chunk_num_lines
        self.total_lines_num = total_lines_num
        self.save_path = Path(save_path).expanduser().resolve()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path).expanduser().resolve()
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_parse_errors = log_parse_errors
        self.log_parse_progress = log_parse_progress
        if os.path.exists(self.log_path / self.log_parse_errors):
            os.remove(self.log_path / self.log_parse_errors)
        if os.path.exists(self.log_path / self.log_parse_progress):
            os.remove(self.log_path / self.log_parse_progress)
        
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

                if "claims" in entity_dict:
                    triplets = []
                    for relation in entity_dict["claims"]:
                        if relation in ["P31", "P279", "P106", "P734"]:
                            objects_list = []
                            objects = entity_dict["claims"][relation]
                            for obj in objects:
                                if "datavalue" in obj["mainsnak"] and "value" in obj["mainsnak"]["datavalue"] and "id" in obj["mainsnak"]["datavalue"]["value"]:
                                    objects_list.append(obj["mainsnak"]["datavalue"]["value"]["id"])
                            if objects_list:
                                triplets.append([relation] + objects_list)
                    entity_info["triplets"] = triplets

                    number_of_relations = len(entity_dict["claims"])
                    entity_info["number_of_relations"] = number_of_relations

        except Exception as e:
            self.log_to_file("process_sample_error", entity_dict, self.log_parse_errors)
            self.log_to_file("process_sample_error", e, self.log_parse_errors)

        return entity_id, entity_info

    def run(self, num_proc, lines_list):
        """
        Method for parallel processing of lines from Wikidata file
        """
        cur_wiki_dict = {}
        for line in lines_list:
            line = line[:-2]
            entity_id = ""
            try:
                entity = json.loads(line)
                entity_id, entity_info = self.process_sample(entity)
            except Exception as e:
                self.log_to_file("run_error", line, self.log_parse_errors)
                self.log_to_file("run_error", e, self.log_parse_errors)
            if entity_id:
                cur_wiki_dict[entity_id] = entity_info
        self.wiki_dict[num_proc] = cur_wiki_dict
                     

    def parse(self, continue_parsing: bool = False):
        """
        Method for parsing of Wikidata file
        """
        line = self.bz_file.readline()

        parsed_lines = 0
        num_iterations = 0
        if continue_parsing:
            files = os.listdir(self.save_path)
            self.log_to_file("already existing files", files, self.log_parse_progress)
            num_iterations = len(files)
            
            total_parsed_elements = 0
            for filename in files:
                with open(f"{self.save_path}/{filename}", 'r') as fl:
                    already_parsed_data = pickle.load(fl)
                    total_parsed_elements += len(already_parsed_data)
            self.log_to_file("total_parsed_elements", total_parsed_elements, self.log_parse_progress)

            for _ in range(total_parsed_elements):
                line = self.bz_file.readline()
                parsed_lines += 1

        while True:
            log.debug(f"iteration number {num_iterations}")
            self.log_to_file("iteration number", num_iterations, self.log_parse_progress)
            self.wiki_dict = self.manager.dict()
            common_list = []

            count = 0
            while line:
                line = self.bz_file.readline()
                common_list.append(line)
                count += 1
                parsed_lines += 1
                if count == self.chunk_num_lines * self.num_processors:
                    log.info(f"parsed lines {round(100*parsed_lines/self.total_lines_num)}")
                    break
            
            if not common_list:
                break

            workers = []
            for ii in range(self.num_processors):
                worker = mp.Process(target=self.run, args=(ii, common_list[ii*self.chunk_num_lines:(ii+1)*self.chunk_num_lines]))
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()

            total_dict = {}
            for key in self.wiki_dict:
                for entity in self.wiki_dict[key]:
                    total_dict[entity] = self.wiki_dict[key][entity]

            save_pickle(total_dict, self.save_path / f"{num_iterations}.pickle")

            num_iterations += 1
            
    def log_to_file(self, log_title, log_info, filename):
        with open(self.log_path / filename, 'a') as out:
            out.write(str(log_title)+'\n')
            out.write("_"*70+'\n')
            out.write(str(log_info)+'\n')
            out.write("_"*70+'\n')
        out.close()
