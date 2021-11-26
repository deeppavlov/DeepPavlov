import datetime
import json
from logging import getLogger
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Optional

import filelock
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from aliases import Aliases
from constants import METRICS_FILENAME, LOCKFILE
from deeppavlov import build_model, deep_download
from deeppavlov.core.data.utils import jsonify_data
from main import initial_setup, redirect_std, download_wikidata, parse_wikidata, parse_entities, update_faiss

logger = getLogger(__file__)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

deep_download('entity_linking_vx_siam_distil.json')
initial_setup()
el_model = build_model('entity_linking_vx_siam_distil.json', download=False)

with open("/data/el_test_samples.json", 'r') as fl:
    init_test_data = json.load(fl)


class Batch(BaseModel):
    entity_substr: List[List[str]] = Field(..., example=[["москва", "россии"]])
    entity_offsets: List[List[List[int]]] = Field(..., example=[[[0, 6], [17, 23]]])
    tags: List[List[str]] = Field(..., example=[["LOC", "LOC"]])
    sentences_offsets: List[List[List[int]]] = Field(..., example=[[[0, 24]]])
    sentences: List[List[str]] = Field(..., example=[["Москва - столица России."]])
    probas: List[List[float]] = Field(..., example=[[0.42]])


@app.post("/model")
async def model(payload: Batch):
    res = el_model(payload.entity_substr,
                   payload.entity_offsets,
                   payload.tags,
                   payload.sentences_offsets,
                   payload.sentences,
                   payload.probas)
    entity_substr, conf, entity_offsets, entity_ids, entity_tags, entity_labels, status = res
    response = {
        "entity_substr": entity_substr,
        "conf": conf,
        "entity_offsets": entity_offsets,
        "entity_ids": entity_ids,
        "entity_tags": entity_tags,
        "entity_labels": entity_labels,
        "status": status
    }
    return jsonify_data(response)


@app.get('/last_train_metric')
async def get_metric():
    if Path(METRICS_FILENAME).exists():
        df = pd.read_csv(METRICS_FILENAME)
        last_metrics = df.iloc[-1].to_dict()
        logger.warning(f"last_metrics {last_metrics}")

        return jsonify_data({"success": True, "data": {"time": str(last_metrics["time"]),
                                          "old_precision": float(last_metrics["old_precision"]),
                                          "new_precision": float(last_metrics["new_precision"]),
                                          "old_recall": float(last_metrics["old_recall"]),
                                          "new_recall": float(last_metrics["new_recall"]),
                                          "update_model": bool(last_metrics["update_model"])}})
    raise HTTPException(status_code=424, detail='Metrics not found. Call /evaluate to evaluate metrics.')


@app.post("/evaluate")
async def model(fl: Optional[UploadFile] = File(None)):
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

        entity_ids_list = entity_ids_batch[0]
        for entity_ids, gold_entity in zip(entity_ids_list, gold_entities):
            if entity_ids[0] != "not in wiki" and entity_ids[0] == gold_entity:
                num_correct += 1
            if entity_ids[0] != "not in wiki":
                num_found += 1
            if gold_entity != "0":
                num_relevant += 1
    cur_precision = num_correct / num_found
    cur_recall = num_correct / num_relevant

    if Path(METRICS_FILENAME).exists():
        df = pd.read_csv(METRICS_FILENAME)
        max_precision = max(df["old_precision"].max(), df["new_precision"].max())
        max_recall = max(df["old_recall"].max(), df["new_recall"].max())
        if cur_precision > max_precision or cur_recall > max_recall:
            df = df.append({"time": datetime.datetime.now(),
                            "old_precision": max_precision,
                            "new_precision": cur_precision,
                            "old_recall": max_recall,
                            "new_recall": cur_recall,
                            "update_model": True}, ignore_index=True)
    else:
        df = pd.DataFrame.from_dict({"time": [datetime.datetime.now()],
                                     "old_precision": [cur_precision],
                                     "new_precision": [cur_precision],
                                     "old_recall": [cur_recall],
                                     "new_recall": [cur_recall],
                                     "update_model": [False]})
    df.to_csv(METRICS_FILENAME, index=False, float_format='%.3f')
    return {"precision": cur_precision, "recall": cur_recall}


def start_process(foo):
    try:
        with FileLock(LOCKFILE, timeout=1):
            p = Process(target=foo)
            p.start()
            status_code, message = 200, 'Process successfully started.'
    except Timeout:
        status_code, message = 409, 'Update is already running.'
    return JSONResponse(status_code=status_code, content={'success': status_code == 200, 'message': message})


@app.get('/update/model')
async def update_model():
    """Starts model update using current parsed wikidata and aliases list"""
    def _update_model():
        with FileLock(LOCKFILE):
            redirect_std()
            parse_entities()
            update_faiss()
        LOCKFILE.unlink()
    return start_process(_update_model)


@app.get('/update/wikidata')
async def update_wikidata():
    """Download wikidata, parse it, update model using new wikidata and aliases"""
    def _update_wikidata():
        with FileLock(LOCKFILE):
            redirect_std()
            download_wikidata()
            parse_wikidata()
            parse_entities()
            update_faiss()
        LOCKFILE.unlink()
    return start_process(_update_wikidata)


@app.get('/status')
async def proba():
    """Returns status of update process.

    Update functions use filelock to prevent starting multiple update processes simultaneously. In the end both
    functions remove lock file. To check update processes status this function checks if lockfile exists,
    either it acquired or not.
    """
    if LOCKFILE.exists():
        try:
            with filelock.FileLock(LOCKFILE, timeout=1):
                message = 'failed'
        except filelock.Timeout:
            message = 'running'
    else:
        message = 'finished sucessfully'
    return JSONResponse(status_code=200, content={'success': True, 'message': message})


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
