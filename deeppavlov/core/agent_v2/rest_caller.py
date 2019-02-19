from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Any, Optional

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

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def __call__(self, names: List[str], urls: List[str], payload: dict) -> List[Dict[str, Dict[str, Any]]]:
        services_functions = [partial(_make_request, name, url)
                              for name, url in zip(names, urls)]
        total_result = []
        for preprocessed in zip(*self.executor.map(lambda f: f(payload), services_functions)):
            res = {}
            for data in preprocessed:
                res.update(data)

            total_result.append(res)

        return total_result
