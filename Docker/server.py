import asyncio
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
from logging import getLogger

import aiohttp
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from aliases import Aliases
from main import initial_setup, State, download_wikidata, parse_wikidata, parse_entities, update_faiss
from deeppavlov import build_model, deep_download
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.data.utils import simple_download

logger = getLogger(__file__)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

metrics_filename = "/data/metrics_score_history.csv"

simple_download("http://files.deeppavlov.ai/rkn_data/el_test_samples.json", "/data/el_test_samples.json")
with open("/data/el_test_samples.json", 'r') as fl:
    init_test_data = json.load(fl)

el_config = parse_config('entity_linking_vx_siam_distil.json')
deep_download('entity_linking_vx_siam_distil.json')
el_model = build_model(el_config, download=False)


@app.post("/model")
async def model(request: Request):
    while True:
        try:
            input_data = await request.json()
            entity_substr = input_data["entity_substr"]
            entity_offsets = input_data["entity_offsets"]
            tags = input_data["tags"]
            sentences_offsets = input_data["sentences_offsets"]
            sentences = input_data["sentences"]
            probas = input_data["probas"]
            res = el_model(entity_substr, entity_offsets, tags, sentences_offsets, sentences, probas)
            entity_substr, conf, entity_offsets, entity_ids, entity_tags, entity_labels, status = res
            response = {"entity_substr": entity_substr, "conf": conf, "entity_offsets": entity_offsets,
                        "entity_ids": entity_ids, "entity_tags": entity_tags, "entity_labels": entity_labels,
                        "status": status}
            return response
        
        except:
            logger.warning(f'Interal server error')


@app.get('/last_train_metric')
async def get_metric(request: Request):
    while True:
        try:
            last_metrics = {}
            if Path(metrics_filename).exists():
                df = pd.read_csv(metrics_filename)
                last_metrics = df.iloc[-1].to_dict()
                logger.warning(f"last_metrics {last_metrics}")
            
            return {"success": True, "data": {"time": str(last_metrics["time"]),
                                              "old_precision": float(last_metrics["old_precision"]),
                                              "new_precision": float(last_metrics["new_precision"]),
                                              "old_recall": float(last_metrics["old_recall"]),
                                              "new_recall": float(last_metrics["new_recall"]),
                                              "update_model": bool(last_metrics["update_model"])}}
            
        except:
            logger.warning(f'Interal server error')


@app.get("/evaluate")
async def model(fl: Optional[UploadFile] = File(None)):
    while True:
        try:
            if fl:
                test_data = json.loads(await fl.read())
            else:
                test_data = init_test_data
            
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
                entity_substr_batch, conf_batch, entity_offsets_batch, entity_ids_batch, entity_tags_batch, \
                    entity_labels_batch, status_batch = el_model([entity_substr], [entity_offsets], [tags],
                                                                 [sentences_offsets], [sentences], [probas])
                
                entity_substr_list = entity_substr_batch[0]
                conf_list = conf_batch[0]
                entity_offsets_list = entity_offsets_batch[0]
                entity_ids_list = entity_ids_batch[0]
                entity_tags_list = entity_tags_batch[0]
                entity_labels_list = entity_labels_batch[0]
                status_list = status_batch[0]
                for entity_ids, gold_entity in zip(entity_ids_list, gold_entities):
                    if entity_ids[0] != "not in wiki" and entity_ids[0] == gold_entity:
                        num_correct += 1
                    if entity_ids[0] != "not in wiki":
                        num_found += 1
                    if gold_entity != "0":
                        num_relevant += 1
            cur_precision = round(num_correct / num_found, 3)
            cur_recall = round(num_correct / num_relevant, 3)
            
            if Path(metrics_filename).exists():
                df = pd.read_csv(metrics_filename)
                max_precision = max(df["old_precision"].max(), df["new_precision"].max())
                max_recall = max(df["old_recall"].max(), df["new_recall"].max())
                if cur_precision > max_precision or cur_recall > max_recall:
                    df = df.append({"time": datetime.datetime.now(),
                                    "old_precision": max_precision,
                                    "new_precision": cur_precision,
                                    "old_recall": max_recall,
                                    "new_recall": cur_recall,
                                    "update_model": False}, ignore_index=True)
            else:
                df = pd.DataFrame.from_dict({"time": [datetime.datetime.now()],
                                             "old_precision": [cur_precision],
                                             "new_precision": [cur_precision],
                                             "old_recall": [cur_recall],
                                             "new_recall": [cur_recall],
                                             "update_model": [False]})
            df.to_csv(metrics_filename, index=False)
            return {"precision": cur_precision, "recall": cur_recall}
        
        except:
            logger.warning(f'Interal server error')

            
@app.get('/last_train_metric')
async def get_metric(request: Request):
    while True:
        try:
            last_metrics = {}
            if Path(metrics_filename).exists():
                df = pd.read_csv(metrics_filename)
                last_metrics = df.iloc[-1].to_dict()
                logger.warning(f"last_metrics {last_metrics}")
            
            return {"success": True, "data": {"time": str(last_metrics.get("time", "")),
                                              "old_precision": float(last_metrics.get("old_precision", "")),
                                              "old_recall": float(last_metrics.get("old_recall", "")),
                                              "new_precision": float(last_metrics.get("new_precision", "")),
                                              "new_recall": float(last_metrics.get("new_recall", "")),
                                              "update_model": bool(last_metrics.get("update_model", ""))}}
            
        except:
            logger.warning(f'Interal server error')


@app.get('/update/wikidata')
async def update_wikidata():
    state = State.from_yaml()
    download_wikidata(state)
    parse_wikidata(state)


@app.get('/update/model')
async def update_model():
    state = State.from_yaml()
    parse_entities(state)
    update_faiss(state)


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


uvicorn.run(app, host='0.0.0.0', port=8000)
'''
{"entity_substr": [["москва", "россии"]],"entity_offsets": [[[0, 6], [17, 23]]],"tags": [["LOC", "LOC"]],"sentences_offsets": [[[0, 24]]],"sentences": [["Москва - столица России."]]}
'''
