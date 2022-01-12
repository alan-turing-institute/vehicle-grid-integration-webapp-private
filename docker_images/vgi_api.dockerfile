FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim


# Install dependencies
RUN apt-get update && \
    apt-get -y install curl 

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/etc/poetry python && \
    cd /usr/local/bin && \
    ln -s /etc/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY vgi_api /app/vgi_api
WORKDIR /app/vgi_api
RUN poetry install --no-root --no-dev