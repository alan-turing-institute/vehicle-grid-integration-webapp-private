# Vehicle Grid Integration

A Frontend (Vue) and REST API (FastAPI) to simulate the effect of electric vehicles, heat pumps, solar photovoltaic and other loads on the electricity grid.

✨ Check out the VGI website at [https://www.e4futuregrid.com/](https://www.e4futuregrid.com/) ✨


## Getting started

### Running locally

#### API/Backend

The VGI API is written in [FastAPI](https://fastapi.tiangolo.com/) and is in the [/vgi_api](/vgi_api/) directory.


[`vgi_api`](vgi_api) is the VGI API (a python package).



## WebApp

## Project setup
```
npm install
```

### Compiles and hot-reloads for development

```
npm run serve
```

You can change the API endpoint by altering [.env.development](.env.development) which defaults to `http://127.0.0.1:8000`.

This assumes you are running the API server locally, which you can do with:

```bash
cd vgi_api && poetry run uvicorn vgi_api:app --reload --port 8000
```

If you want to use the production API (i.e. on Azure) change the contents of [.env.development](.env.development) to match that in [.env.production](.env.production).



### Compiles and minifies for production
```
npm run build
```

### Lints and fixes files
```
npm run lint
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).
