### Prepare host dir for container data
For example, we will use `~/data`. In this directory there are following subderictories and files:
- `downloads` - Directory to download pretrained model data from DeepPavlov file server.
- `parsed_wikidata`, `parsed_wikidata_old`, `parsed_wikidata_new` - Parsed wikidata from downloads is copied to
`parsed_wikidata` directory. When wikidata update is started, new wikidata is parsed to `parsed_wikidata_new`. If
parsing was successfull, `parsed_wikidata_new` renamed to `parsed_wikidata`, `parsed_wikidata_old is deleted` and
`parsed_wikidata` is renamed to `parsed_wikidata_old`.
- `entities`, `entities_old`, `entities_new` - Similar to `parsed_wikidata*` directories.
- `faiss`, `faiss_old`, `faiss_new` - Similar to `parsed_wikidata*` directories.
- `wikidata.json.bz2` - Wikidata file. Downloaded when `/update/wikidata` is called.
- `metrics_score_history.csv` - Table with metrics scores obtained when `/evaluate` is called.
- `aliases.pickle` - File with aliases dictionary.
- `el_test_samples.json` - Default payload for `/evaluate` endpoint.
- `logs` - Directory with log files of update processes.
- `lockfile` - File to synchronize update processes.

### Start container
Don't forget to change host mapping path for container `/data` dir in `docker-compose.yml` from `~/data` to correct one.

```shell
docker-compose up --build
```

### API
All endpoints are for master container
You could open `/docs` in web browser to get Swagger

* POST `/model` - infer model
* GET `/last_train_metric` - get model metrics after last evaluation
* POST `/evaluate` - evaluate model accuracy. Sample is http://files.deeppavlov.ai/rkn_data/el_test_samples.json
* GET `/update/model` - update model if wikidata or aliases list was updated. Raises exception with 409 status code
if process with model or wikidata update is currently running
* GET `/update/wikidata` - download new wikidata, parse it to .pickle files, update model Raises exception with 409 
status code if process with model or wikidata update is currently running
* GET `/status` - returns status of the current update process
* GET `/aliases` - get list of aliases
* GET `/aliases/get/{entity_id}` - get list of aliases for an entity
* POST `/aliases/add/{label}` ["entity_id_1", "entity_id_2"] - add new alias. Example:
`curl -X POST "http://10.11.1.1:8000/aliases/add/%D0%B2%D0%B2%D0%BF" -H  "accept: application/json" -H  "Content-Type: application/json" -d "[\"Q7747\"]"`
* POST `/aliases/add_many` {"label1": ["e1", "e2"], "label2": ["e3", "e4"]} - add many aliases
* GET `/aliases/delete/{label}` - delete alias with label `{label}`

###Logs

See logs from processes started after calling wikidata or model update at `/data/logs` directory.

###Resources
GPU mode: 7731MiB VRAM and 5Gb RAM
CPU mode: 9Gb RAM
