import asyncio
import re
from itertools import cycle
from typing import Dict, Optional

import aiohttp
import docker
import yaml
from aiohttp import client_exceptions
from docker.models.containers import Container
from docker.types import DeviceRequest

from constants import DATA_PATH, HOST_DATA_PATH, CONTAINERS_CONFIG_PATH


class Porter:
    def __init__(self):
        self.__client = docker.from_env()
        with open(CONTAINERS_CONFIG_PATH) as fin:
            self.params = yaml.safe_load(fin)
        self.workers: Dict[str, Container] = {}
        for name, env_vars in self.params.items():
            self.workers[name] = self.start_worker(name, env_vars)
        self.active_hosts = cycle(self.workers)
        self.manager: Optional[Container] = None

    async def update_containers(self):
        for name, env_vars in self.params.items():
            container = self.workers.pop(name)
            self.active_hosts = cycle(self.workers)
            while True:
                requests_in_process = await self.requests_in_process(name)
                if requests_in_process > 0:
                    await asyncio.sleep(1)
                else:
                    break
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, container.restart)
            for i in range(30):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"http://{name}:8000/probe", json={}) as resp:
                            break
                except client_exceptions.ClientConnectorError:
                    await asyncio.sleep(10)
            else:
                raise TimeoutError(f"can't restart a container {name}")
            self.workers[name] = container
            self.active_hosts = cycle(self.workers)

    @staticmethod
    async def requests_in_process(name):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"http://{name}:8000/metrics") as resp:
                    text = await resp.text()
                    match = re.search('http_requests_in_progress{endpoint="/model"} (.*)', text)
                    if match is None:
                        return 0.0
                    else:
                        return float(match.group(1))
            except client_exceptions.ClientConnectorError:
                print(f'container {name} is unavailable. Restarting')
                return 0.0

    def start_worker(self, name: str, env_vars: dict) -> Container:
        env_vars['CONTAINER_NAME'] = name
        return self.__client.containers.run('client',
                                            detach=True,
                                            device_requests=[DeviceRequest(count=-1,
                                                                           capabilities=[['gpu'], ['nvidia'],
                                                                                         ['compute'], ['compat32'],
                                                                                         ['graphics'], ['utility'],
                                                                                         ['video'], ['display']])],
                                            network='el_network',
                                            volumes={str(HOST_DATA_PATH): {'bind': str(DATA_PATH), 'mode': 'rw'}},
                                            name=name,
                                            remove=True,
                                            environment=env_vars)

# container.attrs['State']['Running']