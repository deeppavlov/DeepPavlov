import bz2
import json
import multiprocessing as mp
import os
import unidecode
from pathlib import Path
from collections import defaultdict
from nltk.corpus import stopwords
from deeppavlov.core.data.utils import simple_download
from .wikidata_parse import WikidataParser
from .entities_parse import EntitiesParser

save_path = "~/.deeppavlov/downloads/wikidata"
wikidata_filename = "wikidata.json.bz2"
save_path = Path(os.path.join(save_path, wikidata_filename)).expanduser().resolve()
save_path.parent.mkdir(parents=True, exist_ok=True)
url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
simple_download(url, save_path)

wikidata_parser = WikidataParser(wikidata_filename)
wikidata_parser.parse()

entities_parser = EntitiesParser(load_path="~/.deeppavlov/downloads/wikidata_rus",
                                 save_path="~/.deeppavlov/downloads/wikidata_rus",
                                 name_to_idlist_filename="name_to_idlist_ru.pickle",
                                 ent_list_filename="ent_list_rus.pickle",
                                 word_to_idlist_filename="word_to_idlist_rus.pickle",
                                 entities_ranking_dict_filename="entities_ranking_dict_rus.pickle",
                                 entities_descr_filename="q_to_descr_ru.pickle")

entities_parser.load()
entities_parser.parse()
entities_parser.save()
