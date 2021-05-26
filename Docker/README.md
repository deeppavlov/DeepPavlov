### Build master and worker images:
```shell
docker build -t el_master .
docker build -t client -f worker_dockerfile .
```

### Create network for containers
```shell
docker network create --driver bridge el_network
```

### Prepare host dir for container data
For example, we will use `/data`
#### Create state file:
`/data/state.yaml`:
```yaml 
aliases_updated: 2021-02-07 18:11:30.774029
entities_parsed: '2021-02-09 08:42:59.303435'
entities_wikidata: Wed, 27 Jan 2021 19:42:54 GMT
faiss_updated: '2021-02-09 08:42:59.303435'
wikidata_created: Wed, 27 Jan 2021 19:42:54 GMT
wikidata_parsed: Wed, 27 Jan 2021 19:42:54 GMT
```
#### Create configuration file for containers:
`/data/containers.yaml`:
```yaml
el_worker_0:
    CUDA_VISIBLE_DEVICES:
        0
el_worker_1:
    CUDA_VISIBLE_DEVICES:
        1
```
Keys are names for worker containers, `CUDA_VISIBLE_DEVICES` should be defined according your gpu distribution over
containers. If you want to start worker without GPU, use `''` as `CUDA_VISIBLE_DEVICES` value.

### Start master container
Don't forget to change path <data_dir>. Host port also can be changed
```shell
docker run -v /var/run/docker.sock:/var/run/docker.sock --net=el_network -v <data_dir>:/data -p 8000:8000 -e HOST_DATA_PATH=<data_dir> el_master
```

### API
All endpoints are for master container
You could open `/docs` in web browser to get Swagger

* POST `/model` - infer model
* GET `/update/wikidata` - download new wikidata and parse it to .pickle files
* GET `/update/model` - update model if wikidata or aliases list was updated
* GET `/update/containers` - update all containers with the updated data (which will also lead to workers reload)
* GET `/status` - get status of containers
* GET `/aliases` - get list of aliases
* GET `/aliases/get/{entity_id}` - get list of aliases for an entity
* POST `/aliases/add/{label}` ["entity_id_1", "entity_id_2"] - add new alias. Example:
`curl -X POST "http://10.11.1.1:8000/aliases/add/%D0%B2%D0%B2%D0%BF" -H  "accept: application/json" -H  "Content-Type: application/json" -d "[\"Q7747\"]"`
* POST `/aliases/add_many` {"label1": ["e1", "e2"], "label2": ["e3", "e4"]} - add many aliases
* GET `/aliases/delete/{label}` - delete alias with label `{label}`

###Resources
GPU mode: 7731MiB VRAM and 5Gb RAM
CPU mode: 9Gb RAM
