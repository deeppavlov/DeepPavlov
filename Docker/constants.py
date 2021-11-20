from pathlib import Path
from os import getenv

DATA_PATH = Path(getenv('DATA_PATH', '/data')).resolve()
DOWNLOADS_PATH = Path('/data/downloads').resolve()
HOST_DATA_PATH = Path(getenv('HOST_DATA_PATH', '~/data')).resolve()

WIKIDATA_PATH = DATA_PATH / 'wikidata.json.bz2'
WIKIDATA_URL = 'https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2'

PARSED_WIKIDATA_PATH = DOWNLOADS_PATH / 'parsed_wikidata'
PARSED_WIKIDATA_OLD_PATH = DOWNLOADS_PATH / 'parsed_wikidata_old'
PARSED_WIKIDATA_NEW_PATH = DOWNLOADS_PATH / 'parsed_wikidata_new'

ENTITIES_PATH = DOWNLOADS_PATH / 'entities'
ENTITIES_OLD_PATH = DOWNLOADS_PATH / 'entities_old'
ENTITIES_NEW_PATH = DOWNLOADS_PATH / 'entities_new'

FAISS_PATH = DOWNLOADS_PATH / 'faiss'
FAISS_OLD_PATH = DOWNLOADS_PATH / 'faiss_old'
FAISS_NEW_PATH = DOWNLOADS_PATH / 'faiss_new'

STATE_PATH = DATA_PATH / 'state.yaml'
ALIASES_PATH = ENTITIES_PATH / 'q_to_alias_vx.pickle'
CONTAINERS_CONFIG_PATH = DATA_PATH / 'containers.yaml'
LOGS_PATH = DATA_PATH / 'logs'
