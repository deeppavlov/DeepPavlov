### Build master and worker images:
```shell
docker build --no-cache -t el_master .
```

docker network create --driver bridge el_network

docker run -v /var/run/docker.sock:/var/run/docker.sock --net=el_network -v /home/ignatov/vx:/root/vx -p 8000:8000 asdf