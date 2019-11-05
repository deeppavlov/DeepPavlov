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

import concurrent
from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.api_requester import ApiRequester

logger = getLogger(__name__)


@register("api_router")
class ApiRouter(Component):
    """A helper class for running multiple API requesters on the same data in parallel

    Args:
        api_requesters: list of ApiRequester objects
        n_workers: The maximum number of subprocesses to run

    Attributes:
        api_requesters: list of ApiRequester objects
        n_workers: The maximum number of subprocesses to run
    """

    def __init__(self, api_requesters: List[ApiRequester], n_workers: int = 1, *args, **kwargs):
        self.api_requesters = api_requesters
        self.n_workers = n_workers

    def __call__(self, *args):
        """

        Args:
            *args: list of arguments to forward to the API requesters

        Returns:
            results of the requests
        """
        with ProcessPoolExecutor(self.n_workers) as executor:
            futures = [executor.submit(api_requester, *args) for api_requester
                       in
                       self.api_requesters]

            concurrent.futures.wait(futures)
            results = []
            for future, api_requester in zip(futures, self.api_requesters):
                result = future.result()
                if api_requester.out_count > 1:
                    results += result
                else:
                    results.append(result)

        return results
