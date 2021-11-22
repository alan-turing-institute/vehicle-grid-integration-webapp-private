# VGI API and WebAPp


## API/Backend

The VGI API is written in [FastAPI](https://fastapi.tiangolo.com/) and run in production via [Azure Functions](https://docs.microsoft.com/en-us/azure/azure-functions/).

The project directory contains the following folders
```
vgi_api/
azure_funcs/
```


[`vgi_api`](vgi_api) is the VGI API (a python package).

[`azure_funcs`](azure_funcs) is the Azure Function App which runs `vgi_api` in production.


## WebApp

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

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
