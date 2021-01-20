import os
from pathlib import Path
from deeppavlov.core.data.utils import simple_download
from deeppavlov.models.kbqa.download_parse_utils.wikidata_parse import WikidataParser
from deeppavlov.models.kbqa.download_parse_utils.entities_parse import EntitiesParser


save_path = "~/.deeppavlov/downloads/wikidata"
wikidata_filename = "wikidata.json.bz2"
wikidata_path = Path(os.path.join(save_path, wikidata_filename)).expanduser().resolve()
wikidata_path.parent.mkdir(parents=True, exist_ok=True)

url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"

simple_download(url, wikidata_path)


wikidata_path = "/data/.deeppavlov/downloads/wikidata/latest-all.json.bz2"
wikidata_parser = WikidataParser(wikidata_path,
                                 chunk_num_lines = 6000,
                                 save_path = "~/.deeppavlov/downloads/wikidata_parse")
wikidata_parser.parse()


entities_parser = EntitiesParser(load_path="~/.deeppavlov/downloads/wikidata_parse",
                                 save_path="~/.deeppavlov/downloads/wikidata_rus",
                                 word_to_idlist_filename="word_to_idlist_rus.pickle",
                                 entities_ranking_dict_filename="entities_ranking_dict_rus.pickle",
                                 entities_descr_filename="q_to_descr_ru.pickle")

entities_parser.load()
entities_parser.parse()
entities_parser.save()

