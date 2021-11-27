from filelock import FileLock

from constants import LOCKFILE
from main import parse_entities, update_faiss

with FileLock(LOCKFILE):
    parse_entities()
    update_faiss()
LOCKFILE.unlink()
