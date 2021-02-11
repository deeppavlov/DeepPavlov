### Build master and worker images:
```shell
docker build --no-cache -t el_master .
docker build --no-cache -t client -f worker_dockerfile .
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
Don't forget to change path <data_dir>. Host port also could be changed
```shell
docker run -v /var/run/docker.sock:/var/run/docker.sock --net=el_network -v <data_dir>:/data -p 8000:8000 -e HOST_DATA_PATH=<data_dir> el_master
```