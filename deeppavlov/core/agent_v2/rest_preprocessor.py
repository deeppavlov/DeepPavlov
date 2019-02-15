from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Any, Optional

import requests


def _annotator_request(name, url, payload):
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f'Got {r.status_code} status code for {url}')
    return [{name: response} for response in r.json()['responses']]


class Preprocessor:

    def __init__(self, names: List[str], urls: List[str], *,
                 max_workers: Optional[int] = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.names = names

        self.annotators_functions = [partial(_annotator_request, name, url)
                                     for name, url in zip(names, urls)]

    def __call__(self, payload: dict) -> List[Dict[str, Dict[str, Any]]]:
        annotations = []
        for preprocessed in zip(*self.executor.map(lambda f: f(payload), self.annotators_functions)):
            dialog_annotations = {}
            for data in preprocessed:
                dialog_annotations.update(data)

            annotations.append(dialog_annotations)

        return annotations
