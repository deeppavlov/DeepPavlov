from concurrent.futures import ProcessPoolExecutor
import concurrent

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component

logger = get_logger(__name__)


@register("api_router")
class ApiRouter(Component):

    def __init__(self, api_requesters, n_workers=1, *args, **kwargs):
        self.api_requesters = api_requesters
        self.n_workers = n_workers

    def __call__(self, *args, **kwargs):
        with ProcessPoolExecutor(self.n_workers) as executor:
            futures = [executor.submit(api_requester, *args) for api_requester
                       in
                       self.api_requesters]

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return resultsgit 