## Build and run a docker image of the VGI API

You will need [Docker](https://docs.docker.com/get-docker/) installed.



Build docker image from the repository root directory by running,

```bash
docker build -t vgi_api:latest -f docker_images/vgi_api.dockerfile .   
```

You can then run the docker image with,
```bash
docker run -it -p 8000:80  -e APP_MODULE="vgi_api:app" vgi_api:latest -e NETWORKS_DATA_CONTAINER_READONLY=<networks-data-container-readonly> -e NETWORKS_DATA_CONTAINER_READONLY_CONNECTION_STRING <networks-data-container-readonly-connection-string>
```

where `<networks-data-container-readonly>` and `<networks-data-container-readonly-connection-string>` are described [here](/vgi_api).

The argument `-p 8000:80` means connect port 80 in the docker image to port 8000 on your local machine. You can then run the frontend as described in the project [README](../README.md#WebApp).