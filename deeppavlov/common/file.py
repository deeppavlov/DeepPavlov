import json


def read_json(fpath):
    with open(fpath) as fin:
        return json.load(fin)

