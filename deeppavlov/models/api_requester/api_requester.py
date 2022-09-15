# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Any, List, Dict, AsyncIterable

import requests

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('api_requester')
class ApiRequester(Component):
    """Component for forwarding parameters to APIs

    Args:
        url: url of the API.
        out: count of expected returned values or their names in a chainer.
        param_names: list of parameter names for API requests.
        debatchify: if ``True``, single instances will be sent to the API endpoint instead of batches.

    Attributes:
        url: url of the API.
        out_count: count of expected returned values.
        param_names: list of parameter names for API requests.
        debatchify: if True, single instances will be sent to the API endpoint instead of batches.
    """

    def __init__(self, url: str, out: [int, list], param_names: [list, tuple] = None, debatchify: bool = False,
                 *args, **kwargs):
        self.url = url
        if param_names is None:
            param_names = kwargs.get('in', ())
        self.param_names = param_names
        self.out_count = out if isinstance(out, int) else len(out)
        self.debatchify = debatchify

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        """

        Args:
            *args: list of parameters sent to the API endpoint. Parameter names are taken from self.param_names.
            **kwargs: named parameters to send to the API endpoint. If not empty, args are ignored

        Returns:
            result of the API request(s)
        """
        data = kwargs or dict(zip(self.param_names, args))

        if self.debatchify:
            batch_size = 0
            for v in data.values():
                batch_size = len(v)
                break

            assert batch_size > 0

            async def collect():
                return [j async for j in self.get_async_response(data, batch_size)]

            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(collect())
            if self.out_count > 1:
                response = list(zip(*response))
        else:
            response = requests.post(self.url, json=data).json()

        return response

    async def get_async_response(self, data: dict, batch_size: int) -> AsyncIterable:
        """Helper function for sending requests asynchronously if the API endpoint does not support batching

        Args:
            data: data to be passed to the API endpoint
            batch_size: requests count

        Yields:
            requests results parsed as json
        """
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None,
                requests.post,
                self.url,
                None,
                {k: v[i] for k, v in data.items()}
            )
            for i in range(batch_size)
        ]
        for r in await asyncio.gather(*futures):
            yield r.json()
