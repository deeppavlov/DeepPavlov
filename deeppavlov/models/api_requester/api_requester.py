import requests

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('api_requester')
class ApiRequester(Component):
    def __init__(self, url: str, out: [int, list], param_names=(), debatchify=False, *args, **kwargs):
        self.url = url
        self.param_names = param_names
        self.out_count = out if isinstance(out, int) else len(out)
        self.debatchify = debatchify

    def __call__(self, *args, **kwargs):
        data = kwargs or dict(zip(self.param_names, args))

        if self.debatchify:
            batch_size = 0
            for v in data.values():
                batch_size = len(v)
                break
            response = [requests.post(self.url, json={k: v[i] for k, v in data.items()}).json()
                        for i in range(batch_size)]
        else:
            response = requests.post(self.url, json=data).json()

        if self.out_count > 1:
            response = list(zip(*response))

        return response
