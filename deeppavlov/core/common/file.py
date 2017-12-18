import json
import pickle


def read_json(fpath):
    with open(fpath) as fin:
        return json.load(fin)


def save_pickle(data, fpath):
    with open(fpath, 'wb') as fout:
        pickle.dump(data, fout)


def load_pickle(fpath):
    with open(fpath, 'rb') as fin:
        return pickle.load(fin)
