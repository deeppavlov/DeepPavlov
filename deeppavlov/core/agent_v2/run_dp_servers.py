import re
from multiprocessing import Process

from deeppavlov.core.agent_v2.config import ANNOTATORS, SKILL_SELECTORS, SKILLS, RESPONSE_SELECTORS
from utils.server_utils.server import skill_server


pattern = re.compile(r'^https?://(?P<host>.*):(?P<port>\d*)(?P<endpoint>.*)$')

processes = []
for item in SKILLS + ANNOTATORS + SKILL_SELECTORS + RESPONSE_SELECTORS:
    if item['path'] is None:
        continue
    url = item['url']
    parsed = pattern.search(url)
    if not parsed:
        raise RuntimeError(f'could not parse an url: `{url}`')
    host, port, endpoint = parsed.groups()
    p = Process(target=skill_server, kwargs={'config': item['path'], 'port': port, 'endpoint': endpoint,
                                             'download': True})
    p.start()
    processes.append(p)

for p in processes:
    p.join()
