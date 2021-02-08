import asyncio
from itertools import cycle
from typing import Dict, Optional

import docker
import yaml
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
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, container.stop)
            await asyncio.sleep(5)
            self.workers[name] = self.start_worker(name, env_vars)
            self.active_hosts = cycle(self.workers)

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
