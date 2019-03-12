from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Any, Optional, Sequence, Union

import requests


def _make_request(name, url, payload):
    r = requests.post(url, json=payload)
    if r.status_code != 200:
        raise RuntimeError(f'Got {r.status_code} status code for {url}')
    return [{name: response} for response in r.json()['responses']]


class RestCaller:
    """
    Call to REST services, annotations or skills.
    """

    def __init__(self, max_workers: Optional[int] = None, names: Optional[Sequence[str]] = None,
                 urls: Optional[Sequence[str]] = None) -> None:
        self.names = tuple(names or ())
        self.urls = tuple(urls or ())
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __call__(self, payload: Union[Dict, Sequence[Dict]], names: Optional[Sequence[Sequence[str]]] = None,
                 urls: Optional[Sequence[Sequence[str]]] = None) -> List[Dict[str, Dict[str, Any]]]:
        if names is None:
            names = self.names
        if urls is None:
            urls = self.urls

        if not isinstance(payload, Sequence):
            payload = [payload] * len(names)

        total_result = []
        for preprocessed in zip(*self.executor.map(_make_request, names, urls, payload)):
            res = {}
            for data in preprocessed:
                res.update(data)

            total_result.append(res)

        return total_result
