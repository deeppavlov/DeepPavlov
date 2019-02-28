from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Collection, Dict, List, Any, Optional, Union, Callable
from functools import partial

from deeppavlov.core.common.chainer import Chainer


class Preprocessor:

    def __init__(self, annotators: Dict[Callable, Union[str, List[Optional[str]]]], *,
                 max_workers: Optional[int] = None) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.annotators = []
        self.keys = []
        for preprocessor, keys in annotators.items():
            if isinstance(keys, str):
                keys = [keys]

            old_preprocessor = preprocessor

            if len(keys) == 1:
                def preprocessor(x, p):
                    return [p(x)]

            elif None in keys:
                indexes, keys = zip(*[(i, k) for i, k in enumerate(keys) if k is not None])
                old_preprocessor = preprocessor

                def preprocessor(x, p):
                    res = p(x)
                    return [res[i] for i in indexes]

            self.keys.extend([k.split('.') for k in keys])
            if isinstance(preprocessor, Chainer):
                self.annotators.append(preprocessor)
            else:
                self.annotators.append(partial(preprocessor, p=old_preprocessor))

    def __call__(self, states: dict) -> List[Dict[str, Any]]:
        annotations = []
        utterances = [dialog['utterances'][-1]['text'] for dialog in states['dialogs']]
        for preprocessed in zip(*chain(*self.executor.map(lambda f: f(utterances), self.annotators))):
            dialog_annotations = {}
            for k, data in zip(self.keys, preprocessed):
                target = dialog_annotations
                for prefix in k[:-1]:
                    target = target.setdefault(prefix, {})
                target[k[-1]] = data

            annotations.append(dialog_annotations)

        return annotations
