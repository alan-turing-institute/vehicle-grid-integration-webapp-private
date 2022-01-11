
   
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim


# Install dependencies
RUN apt-get update && \
    apt-get -y install curl 
#     build-essential \
#     # libcairo2 \
#     libpango* \
#     # libpangocairo-1.0-0 \
#     # libgdk-pixbuf2.0-0 \
#     libffi-dev \
#     shared-mime-info \
#     libpq-dev \
#     python3-dev \
#     git \
#     wget \
#     unzip && \
#     pip install psycopg2 && \
#     rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/etc/poetry python && \
    cd /usr/local/bin && \
    ln -s /etc/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY vgi_api /app/vgi_api
WORKDIR /app/vgi_api
RUN poetry install --no-root --no-dev

# Don't run as root
# RUN groupadd -g 999 appuser && \
#     useradd -r -u 999 -g appuser appuser
# USER appuser