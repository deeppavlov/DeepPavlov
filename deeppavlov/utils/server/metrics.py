# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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

import time
from typing import Tuple

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUESTS_COUNT = Counter('http_requests_count', 'Number of processed requests', ['endpoint', 'status_code'])
REQUESTS_LATENCY = Histogram('http_requests_latency_seconds', 'Request latency histogram', ['endpoint'])
REQUESTS_IN_PROGRESS = Gauge('http_requests_in_progress', 'Number of requests currently being processed', ['endpoint'])


def metrics(request: Request) -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, ignore_paths: Tuple = ()) -> None:
        super().__init__(app)
        self.ignore_paths = ignore_paths

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        endpoint = request.url.path

        if endpoint in self.ignore_paths:
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()

        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            if status_code == 200:
                duration = time.perf_counter() - start_time
                REQUESTS_LATENCY.labels(endpoint=endpoint).observe(duration)
            REQUESTS_COUNT.labels(endpoint=endpoint, status_code=status_code).inc()
            REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()

        return response
