## VGI_API


## Development mode

Ensure you are in the directory containing this README.



To run the API locally first install [Poetry](https://python-poetry.org/docs/), which is a tool for Python packaging and dependency management.

Poetry creates a virtual environment for you and installs all the package dependencies within it.

To install the package and development dependencies run:

```bash
poetry install
```

### Set environment variables

The API reads files from blob storage which were uploaded from the [vehicle-grid-integration-opendss-networks](https://github.com/alan-turing-institute/vehicle-grid-integration-opendss-networks) repo. To access the blob storage we need to set two environment variables:


| Variable                                           | Info                                                                     |
|----------------------------------------------------|--------------------------------------------------------------------------|
| NETWORKS_DATA_CONTAINER_READONLY_CONNECTION_STRING | A connection string/ SAS token used to authenticate against blob storage |
| NETWORKS_DATA_CONTAINER_READONLY                   | The name of the blob storage container                                   |

You can also create a file called `.env` within this directory to store these variables.

### Run the development server

```bash
poetry run uvicorn vgi_api:app --reload --port 8000
```

The `--reload` flag will restart the server whenever you alter code.

### Send a request

Send a request to the API using [HTTPie](https://httpie.io/docs) (you installed it with poetry; or any other http client)

```bash
http POST :8000/simulate
```

### Run as an Azure function

See the instructions in [azure_funcs](../azure_funcs/README.md)


### Run unit tests

```bash
poetry run pytest tests
```
