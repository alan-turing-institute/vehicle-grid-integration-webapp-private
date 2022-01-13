# Vehicle Grid Integration

✨ Check out the deployed version of the VGI website at [https://www.e4futuregrid.com/](https://www.e4futuregrid.com/) ✨


A Frontend (Vue) and REST API (FastAPI) to simulate the effect of electric vehicles, heat pumps, solar photovoltaic and other loads on the electricity grid.

## 💻 Getting started 💻


### API/Backend

The VGI API is written in [FastAPI](https://fastapi.tiangolo.com/) and deployed in production as an [Azure Web App](https://azure.microsoft.com/en-gb/services/app-service/web/). We use [Github Actions](.github/workflows/deploy_azurewebapp_api.yaml) to build a [docker image](/docker_images) and deploy it to Azure whenever a PR is merged to `main`. 

Get started with one of these:

- [Run locally using a development server](/vgi_api): Best for development work.
- [Build a docker image and run locally](/docker_images): Good for testing in an environment similar to production.
- [Explore the continuous delivery pipeline](.github/workflows/deploy_azurewebapp_api.yaml): See how the API is deployed.

### WebApp

To start the WebApp first install make sure you have [npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

You can then install all the project dependencies with,
```
npm install
```

To start a development server run,

```
npm run serve
```

You can change the API endpoint by altering [.env.development](.env.development) which defaults to `http://127.0.0.1:8000`. Port `8000` is the port that the API getting started [instructions](vgi_api) uses.

This assumes you are running the API server locally, which you can do with:

```bash
cd vgi_api && poetry run uvicorn vgi_api:app --reload --port 8000
```

If you want to use the production API (i.e. deployed on Azure) change the contents of [.env.development](.env.development) to match that in [.env.production](.env.production).