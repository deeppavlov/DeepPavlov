import asyncio
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, List

import aiohttp
import docker
import uvicorn
import yaml
from docker.models.containers import Container
from docker.types import DeviceRequest
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from deeppavlov.utils.server.metrics import PrometheusMiddleware
from main import ENTITIES_PATH, FAISS_PATH
from main import Labels
from pydantic import BaseModel

CONTAINERS_CONFIG_PATH = Path('~/vx/containers.yml').expanduser().resolve()
print(CONTAINERS_CONFIG_PATH)
app = FastAPI()

app.add_middleware(
    PrometheusMiddleware,
    ignore_paths=('/', '/metrics', '/api', '/probe', '/docs', '/openapi.json')
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

class Porter:
    def __init__(self):
        with open(CONTAINERS_CONFIG_PATH) as fin:
            self.params = yaml.safe_load(fin)

        self.workers: Dict[str, Container] = {}
        for name, env_vars in self.params.items():
            container = client.containers.run('client',
                                              detach=True,
                                              device_requests=[DeviceRequest(count=-1, capabilities=[['gpu'], ['nvidia'], ['compute'], ['compat32'], ['graphics'], ['utility'], ['video'], ['display']])],
                                              network='el_network',
                                              volumes={'/home/ignatov/vx/parsed': {'bind': str(ENTITIES_PATH), 'mode': 'rw'},
                                                       '/home/ignatov/vx/faiss': {'bind': str(FAISS_PATH), 'mode': 'rw'},
                                                       '/home/ignatov/.deeppavlov': {'bind': '/root/.deeppavlov', 'mode': 'rw'},
                                                       '/home/ignatov/logs': {'bind': '/logs', 'mode': 'rw'}},
                                              name=name,
                                              remove=True,
                                              environment=env_vars)
            self.workers[name] = container
        self.active_hosts = cycle(self.workers)
        self.manager: Optional[Container] = None

    async def update_containers(self):
        for name, env_vars in self.params.items():
            container = self.workers.pop(name)
            self.active_hosts = cycle(self.workers)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, container.stop)
            await asyncio.sleep(1)
            container = client.containers.run('client',
                                              detach=True,
                                              device_requests=[DeviceRequest(count=-1, capabilities=[['gpu'], ['nvidia'], ['compute'], ['compat32'], ['graphics'], ['utility'], ['video'], ['display']])],
                                              network='el_network',
                                              volumes={'/home/ignatov/vx/parsed': {'bind': str(ENTITIES_PATH), 'mode': 'rw'},
                                                       '/home/ignatov/vx/faiss': {'bind': str(FAISS_PATH), 'mode': 'rw'},
                                                       '/home/ignatov/.deeppavlov': {'bind': '/root/.deeppavlov', 'mode': 'rw'},
                                                       '/home/ignatov/logs': {'bind': '/logs', 'mode': 'rw'}},
                                              name=name,
                                              remove=True,
                                              environment=env_vars)
            self.workers[name] = container
            self.active_hosts = cycle(self.workers)

client = docker.from_env()

porter = Porter()


@app.post("/model")
async def model(request: Request):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{next(porter.active_hosts)}:8000/model", json=await request.json()) as resp:
            return await resp.json()


@app.get('/update_containers')
async def update():
    loop = asyncio.get_event_loop()
    loop.create_task(porter.update_containers())
    return 'OK'


@app.get('/labels')
async def get_labels():
    labels = Labels()
    return labels.labels


class Entity(BaseModel):
    label: str
    entity_ids: List[str]


@app.post('/labels')
async def add_label(entity: Entity):
    labels = Labels()
    labels.add_label(entity.label, entity.entity_ids)
    labels.save()
    return 200


@app.get('/worker/{worker_id}')
async def container_logs(worker_id: str):
    if worker_id not in porter.workers:
        return f'no such container'
    else:
        loop = asyncio.get_event_loop()
        return str(await loop.run_in_executor(None, porter.workers[worker_id].logs))

uvicorn.run(app, host='0.0.0.0', port=8000)
'''
{"entity_substr": [["москва", "россии"]],"entity_offsets": [[[0, 6], [17, 23]]],"tags": [["LOC", "LOC"]],"sentences_offsets": [[[0, 24]]],"sentences": [["Москва - столица России."]]}
'''