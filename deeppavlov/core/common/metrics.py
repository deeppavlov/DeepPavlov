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

# Original: https://github.com/perdy/starlette-prometheus

import time

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.types import ASGIApp
from typing import Tuple


REQUESTS = Counter(
    "starlette_requests_total",
    "Total count of requests by method and path.",
    ["method", "path_template"]
)
RESPONSES = Counter(
    "starlette_responses_total",
    "Total count of responses by method, path and status codes.",
    ["method", "path_template", "status_code"],
)
REQUESTS_PROCESSING_TIME = Histogram(
    "starlette_requests_processing_time_seconds",
    "Histogram of requests processing time by path (in seconds)",
    ["method", "path_template"],
)
EXCEPTIONS = Counter(
    "starlette_exceptions_total",
    "Total count of exceptions raised by path and exception type",
    ["method", "path_template", "exception_type"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "starlette_requests_in_progress",
    "Gauge of requests by method and path currently being processed",
    ["method", "path_template"],
)


def metrics(request: Request) -> Response:
    registry = REGISTRY
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        ignore_paths: Tuple = (),
        filter_unhandled_paths: bool = False
    ) -> None:
        super().__init__(app)
        self.filter_unhandled_paths = filter_unhandled_paths
        self.ignore_paths = ignore_paths

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        method = request.method
        path_template, is_handled_path = self.get_path_template(request)

        if self._is_path_ignored(path_template):
            return await call_next(request)

        if self._is_path_filtered(is_handled_path):
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(
            method=method,
            path_template=path_template
        ).inc()
        REQUESTS.labels(method=method, path_template=path_template).inc()
        try:
            before_time = time.perf_counter()
            response = await call_next(request)
            after_time = time.perf_counter()
        except Exception as e:
            EXCEPTIONS.labels(
                method=method,
                path_template=path_template,
                exception_type=type(e).__name__
            ).inc()
            raise e from None
        else:
            REQUESTS_PROCESSING_TIME.labels(
                method=method,
                path_template=path_template
            ).observe(after_time - before_time)
            RESPONSES.labels(
                method=method,
                path_template=path_template,
                status_code=response.status_code
            ).inc()
        finally:
            REQUESTS_IN_PROGRESS.labels(
                method=method,
                path_template=path_template
            ).dec()

        return response

    @staticmethod
    def get_path_template(request: Request) -> Tuple[str, bool]:
        for route in request.app.routes:
            match, child_scope = route.matches(request.scope)
            if match == Match.FULL:
                return route.path, True

        return request.url.path, False

    def _is_path_filtered(self, is_handled_path: bool) -> bool:
        return self.filter_unhandled_paths and not is_handled_path

    def _is_path_ignored(self, path_template: str) -> bool:
        return path_template in self.ignore_paths
