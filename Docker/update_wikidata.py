from filelock import FileLock

from constants import LOCKFILE
from main import download_wikidata, parse_wikidata, parse_entities, update_faiss

with FileLock(LOCKFILE):
    download_wikidata()
    parse_wikidata()
    parse_entities()
    update_faiss()
LOCKFILE.unlink()
