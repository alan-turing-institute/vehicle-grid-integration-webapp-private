import logging
import base64
from . import azure_mockup
from . import azureOptsXmpls as aox

from tempfile import NamedTemporaryFile, TemporaryDirectory
import shutil
import os
import csv

from fastapi import Form, File, UploadFile, HTTPException
from starlette import status

import fastapi
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI()


def save_uploaded_file(file, save_name, size_limit):
    real_file_size = 0
    temp = NamedTemporaryFile(delete=False)
    for chunk in file.file:
        real_file_size += len(
            chunk
        )  # Chunk size (default 1MB) set by starlette https://github.com/encode/starlette/blob/master/starlette/datastructures.py#L412
        if real_file_size > size_limit:
            logging.info(
                "File has reached size {} > limit {}".format(real_file_size, size_limit)
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="{} is too large".format(file.filename),
            )
        temp.write(chunk)
    temp.close()
    shutil.move(temp.name, save_name)


@app.post("/simulate")
async def simulate(
    n_lv: int = Form(5, title="Number of LV networks (up to 20)", ge=0, le=20),
    n_id: int = Form(1060, title="Network ID"),
    c_load: UploadFile = File(None, title="Custom load values"),
):

    logging.info("Starting API call")
    file_name = None

    if c_load is None:
        logging.info("No input file provided")
    else:
        with TemporaryDirectory(prefix="tmp_vgi_uploads_") as tmp_dir:
            tmp_file_name = os.path.join(tmp_dir, "vgi_loads_test.csv")
            save_uploaded_file(c_load, tmp_file_name, 0.1e6)
            file_name = c_load.filename
            logging.info("File name: {}".format(c_load.filename))
            logging.info("File type: {}".format(c_load.content_type))
            logging.info("File saved to {}; contents:".format(tmp_file_name))
            with open(tmp_file_name) as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader)
                l1 = next(reader)
                logging.info(header[:5])
                logging.info(l1[:5])

    parameters = aox.run_dict0
    parameters["network_data"]["n_id"] = n_id
    parameters["network_data"]["n_lv"] = n_lv

    fig1, fig2 = azure_mockup.run_dss_simulation(parameters)
    resultdict = {
        "parameters": parameters,
        "filename": file_name,
        "plot1": base64.b64encode(fig1.getvalue()).decode("utf-8"),
        "plot2": base64.b64encode(fig2.getvalue()).decode("utf-8"),
    }

    return resultdict
