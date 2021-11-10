import asyncio
from typing import Dict, List
from logging import getLogger

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from aliases import Aliases
from porter import Porter

logger = getLogger(__file__)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


porter = Porter()


@app.post("/model")
async def model(request: Request):
    while True:
        try:
            host = next(porter.active_hosts)
        except StopIteration:
            raise HTTPException(status_code=500, detail='No active workers')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{host}:8000/model", json=await request.json()) as resp:
                    return await resp.json(content_type=None)
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))

@app.post("/metrics")
async def model(request: Request):
    while True:
        try:
            host = next(porter.active_hosts)
        except StopIteration:
            raise HTTPException(status_code=500, detail='No active workers')
        try:
            filename_data = await request.json()
            test_filename = filename_data["test_filename"]
            with open(test_filename, 'r') as fl:
                test_data = json.load(fl)
            
            num_correct = 0
            num_found = 0
            num_relevant = 0
            
            for sample in test_data:
                entity_substr = sample["entity_substr"]
                entity_offsets = sample["entity_offsets"]
                tags = sample["tags"]
                probas = sample["probas"]
                sentences = sample["sentences"]
                sentences_offsets = sample["sentences_offsets"]
                gold_entities = sample["gold_entities"]
                res = requests.post(f"http://{host}:8000/model", json={"entity_substr": entity_substr,
                                                                       "entity_offsets": entity_offsets,
                                                                       "tags": tags,
                                                                       "sentences_offsets": sentences_offsets,
                                                                       "sentences": sentences,
                                                                       "probas": probas}).json()
                entity_substr_batch, conf_batch, entity_offsets_batch, entity_ids_batch, entity_tags_batch, \
                    entity_labels_batch, status_batch = res
                for entity_ids, gold_entity in zip(entity_ids_batch[0], gold_entities):
                    if entity_ids[0] != "not in wiki" and entity_ids[0] == gold_entity:
                        num_correct += 1
                    if entity_ids[0] != "not in wiki":
                        num_found += 1
                    if gold_entity != "0":
                        num_relevant += 1
                precision = round(num_correct / num_found, 3)
                recall = round(num_correct, num_relevant, 3)
                return {"precision": precision, "recall": recall}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://{host}:8000/model", json=await request.json()) as resp:
                    return await resp.json(content_type=None)
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))

@app.get('/update/containers')
async def update():
    loop = asyncio.get_event_loop()
    loop.create_task(porter.update_containers())

@app.get('/update/wikidata')
async def update_wikidata():
    try:
        porter.start_manager('python update_wikidata.py')
    except RuntimeError as e:
        return HTTPException(repr(e))

@app.get('/update/model')
async def update_model():
    try:
        porter.start_manager('python update_model.py')
    except RuntimeError as e:
        return HTTPException(repr(e))

@app.get('/aliases')
async def get_aliases():
    return Aliases().aliases


@app.post('/aliases/add/{label}')
async def add_alias(label: str, entity_ids: List[str]):
    aliases = Aliases()
    aliases.add_alias(label, entity_ids)


@app.post('/aliases/add_many')
async def add_alias(new_aliases: Dict[str, List[str]]):
    aliases = Aliases()
    aliases.add_aliases(new_aliases)


@app.get('/aliases/delete/{label}')
async def add_alias(label: str):
    aliases = Aliases()
    if label not in aliases.aliases:
        raise HTTPException(status_code=404, detail=f'Alias with label "{label}" not found')
    aliases.delete_alias(label)
    
    
@app.get('/aliases/get/{label}')
async def get_alias(label: str):
    aliases = Aliases()
    found_aliases = aliases.get_alias(label)
    return f"{found_aliases}"


@app.get('/worker/{worker_id}')
async def container_logs(worker_id: str):
    if worker_id not in porter.workers:
        return f'no such container'
    else:
        loop = asyncio.get_event_loop()
        return str(await loop.run_in_executor(None, porter.workers[worker_id].logs))

@app.get('/status', response_class=HTMLResponse)
async def status():
    containers = await porter.get_stats()
    workers = '\n'.join([f"<tr><td><a href='/logs/{name}'>{name}</a></td><td>{status}</td></tr>" for name, status in containers.items() if name != 'manager'])
    manager = f"<tr><td><a href='/logs/manager'>manager</a></td><td>{containers['manager']}</td></tr>"
    return f"""
    <html>
        <head>
            <title>Entity linking containers</title>
        </head>
        <body>
            <h4>Manager</h4>
            <table>
            <tr><td>name</td><td>status</td></tr>
            {manager}
            </table>
            <h4>Workers</h4>
            <table>
            <tr><td>name</td><td>status</td></tr>
            {workers}
            </table>
        </body>
    </html>
    """

@app.get('/logs/{container_name}', response_class=HTMLResponse)
async def container_logs(container_name: str):
    logs = await porter.get_logs(container_name)
    logs = logs.replace("\n", "<br />")
    return f"""
    <html>
        <head>
            <title>{container_name} logs</title>
        </head>
        <body>
            {logs}
        </body>
    </html>
    """

uvicorn.run(app, host='0.0.0.0', port=8000)
'''
{"entity_substr": [["москва", "россии"]],"entity_offsets": [[[0, 6], [17, 23]]],"tags": [["LOC", "LOC"]],"sentences_offsets": [[[0, 24]]],"sentences": [["Москва - столица России."]]}
'''
