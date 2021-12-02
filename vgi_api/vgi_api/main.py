import logging
import base64

from pydantic.main import BaseModel
from . import azure_mockup
from . import azureOptsXmpls as aox
from typing import Optional, List, Dict
from tempfile import NamedTemporaryFile, TemporaryDirectory
import shutil
import os
import csv
from fastapi.exceptions import RequestValidationError
from fastapi import Query, Form, File, UploadFile, HTTPException
from starlette import status
from enum import Enum
import fastapi
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI()

origins = ["http://localhost:8080", "http://192.168.1.63:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class NetworkID(str, Enum):
    # See https://github.com/alan-turing-institute/e4Future-collab/blob/e1b9430c594e33736c924eaf0793ef484bd146d7/mtd-dss-digital-twins/slesNtwk_turing.py#L212
    URBAN = "1060"
    RURAL = "1061"


class DefaultLV(str, Enum):

    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"


class MVSolarPVOptions(str, Enum):

    NONE = "None"
    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"
    CSV = "csv"


class MVEVChargerOptions(str, Enum):

    NONE = "None"
    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"
    CSV = "csv"


class LVSmartMeterOptions(str, Enum):
    NONE = "None"
    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"
    CSV = "csv"


class LVElectricVehicleOptions(str, Enum):
    NONE = "None"
    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"
    CSV = "csv"


class LVPVOptions(str, Enum):
    NONE = "None"
    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"
    CSV = "csv"


class LVHPOptions(str, Enum):
    NONE = "None"
    OPTION1 = "1"
    OPTION2 = "2"
    OPTION3 = "3"
    CSV = "csv"


class ProfileUnits(str, Enum):

    KW = "kW"
    KWH = "kWh"


# ToDo: Matt to give us full list of options
VALID_LV_NETWORKS_URBAN: List[int] = []
VALID_LV_NETWORKS_RURAL: List[int] = []

DEFAULT_LV_NETWORKS: Dict[NetworkID, Dict[DefaultLV, List[int]]] = {
    NetworkID.RURAL: {
        DefaultLV.OPTION1: [],
        DefaultLV.OPTION2: [],
        DefaultLV.OPTION3: [],
    },
    NetworkID.URBAN: {
        DefaultLV.OPTION1: [],
        DefaultLV.OPTION2: [],
        DefaultLV.OPTION3: [],
    },
}


class LVNetworks(BaseModel):

    networks: List[int]


def validate_lv_parameters(
    lv_list: Optional[str], lv_default: Optional[DefaultLV], n_id: NetworkID
) -> List[int]:
    """Validate the Low Voltage Network parameters and return a list of Low Voltage
    network ids.

    Pass either `lv_list` or `lv_default`. If both are passed will use `lv_list`.

    Args:
        lv_list (Optional[str]): A str of comma seperated network ids
        lv_default (Optional[DefaultLV]): A DefaultLV choice
        n_id (NetworkID): The choice of medium voltage network

    Returns:
        List[int]: [description]
    """

    def _validate_lv_list(
        lv_list: Optional[str], n_id: NetworkID
    ) -> Optional[List[int]]:

        if not lv_list:
            return

        validation_error = HTTPException(
            status_code=422, detail="lv_list values are not valid"
        )

        lv_list_int = [int(i) for i in lv_list.split("")]

        # if len(lv_list_int) > 5:
        #     raise validation_error

        valid = False
        if n_id == NetworkID.URBAN:
            valid = set(lv_list_int).issubset(set(VALID_LV_NETWORKS_URBAN))
        elif n_id == NetworkID.RURAL:
            valid = set(lv_list_int).issubset(set(VALID_LV_NETWORKS_RURAL))

        if not valid:
            raise validation_error

        return lv_list_int

    def _get_default_list(lv_default: DefaultLV, n_id: NetworkID) -> List[int]:

        return DEFAULT_LV_NETWORKS[n_id][lv_default]

    if not (lv_list or lv_default):
        raise HTTPException(
            status_code=422, detail="One of lv_list or lv_default must be provided"
        )

    # If lv list validate the list, otherwise must have selected one of three default lists
    lv_list = _validate_lv_list(lv_list, n_id)

    if not lv_list:
        lv_list = _get_default_list(lv_default, n_id)

    return lv_list


@app.get("/lv-network", response_model=LVNetworks)
async def lv_network(
    n_id: NetworkID = Query(
        ...,
        title="Network ID",
        description="Choice of 11 kV integrated MV-LV network",
    ),
):
    """Return a list of valid LV networks"""

    if n_id == NetworkID.URBAN:
        networks = VALID_LV_NETWORKS_URBAN
    else:
        networks = VALID_LV_NETWORKS_RURAL

    return {"networks": networks}


@app.post("/simulate")
async def simulate(
    n_id: NetworkID = Query(
        ...,
        title="Network ID",
        description="Choice of 11 kV integrated Medium-Low Voltage network",
    ),
    xfmr_scale: float = Query(
        1.0,
        ge=0,
        description="Medium Voltage transformer scaling",
    ),
    oltc_setpoint: float = Query(
        1.0,
        ge=0,
        le=1,
        description="Medium Voltage transformer on-load tap charger (OLTC) set point. Change the set point (in % pu) of the oltc",
    ),
    oltc_bandwidth: float = Query(
        1.0,
        ge=0,
        le=1,
        description="Change the bandwidth (in % pu) of the oltc",
    ),
    rs_pen: float = Query(
        0.8,
        ge=0,
        le=1,
        description="Percentage residential loads",
    ),
    # ToDo: add sensible regex
    lv_list: Optional[str] = Query(
        None,
        help="blah",
        description="Provide a list of up to 5 Low Voltage Network ids. If not provided you must select an option from `lv_default`",
        regex="(\d{4}, ){0,4}\d{4}$",
    ),
    lv_default: Optional[DefaultLV] = Query(
        None, description="Choose a default set of Low Voltage Networks"
    ),
    mv_solar_pv_profile: MVSolarPVOptions = Query(
        MVSolarPVOptions.NONE,
        description="Select a example solar pv profile or select CSV to upload your own. If CSV selected you must provide `mv_solar_pv_csv`",
    ),
    mv_solar_pv_csv: Optional[UploadFile] = File(
        None, description="11kV connected solar photovoltaics (PV)"
    ),
    mv_solar_pv_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="If `mv_solar_pv_csv` provided gives the units",
    ),
    mv_ev_charger_profile: MVEVChargerOptions = Query(
        MVEVChargerOptions.NONE,
        description="Select a example ev profile or select CSV to upload your own. If CSV selected you must provide `mv_ev_charger_csv`",
    ),
    mv_ev_charger_csv: Optional[UploadFile] = File(
        None, title="11kV connected EV fast chargers' stations"
    ),
    mv_ev_charger_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="If `mv_ev_charger` provided gives the units",
    ),
    lv_smart_meter_profile: LVSmartMeterOptions = Query(
        LVSmartMeterOptions.NONE,
        description="",
    ),
    lv_smart_meter_csv: Optional[UploadFile] = File(None, title=""),
    lv_smart_meter_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_electric_vehicle_csv: Optional[UploadFile] = File(None, title=""),
    lv_electric_vehicle_profile: LVElectricVehicleOptions = Query(
        LVElectricVehicleOptions.NONE,
        description="",
    ),
    lv_electric_vehicle_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_pv_csv: Optional[UploadFile] = File(None, title=""),
    lv_pv_profile: LVPVOptions = Query(
        LVPVOptions.NONE,
        description="",
    ),
    lv_pv_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_hp_csv: Optional[UploadFile] = File(None, title=""),
    lv_hp_profile: LVHPOptions = Query(
        LVHPOptions.NONE,
        description="",
    ),
    lv_hp_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
):

    # MV parameters are already valid. LV parameters need additional validation
    validate_lv_parameters(lv_list, lv_default, n_id)

    # ToDo add penetration for EC, PV and HP
    # ToDo: Validate any uploaded files
    # Write the file to disk in a temporary directory

    logging.info("Starting API call")
    file_name = None

    # if c_load is None:
    #     logging.info("No input file provided")
    # else:
    #     with TemporaryDirectory(prefix="tmp_vgi_uploads_") as tmp_dir:
    #         tmp_file_name = os.path.join(tmp_dir, "vgi_loads_test.csv")
    #         save_uploaded_file(c_load, tmp_file_name, 0.1e6)
    #         file_name = c_load.filename
    #         logging.info("File name: {}".format(c_load.filename))
    #         logging.info("File type: {}".format(c_load.content_type))
    #         logging.info("File saved to {}; contents:".format(tmp_file_name))
    #         with open(tmp_file_name) as csv_file:
    #             reader = csv.reader(csv_file)
    #             header = next(reader)
    #             l1 = next(reader)
    #             logging.info(header[:5])
    #             logging.info(l1[:5])

    parameters = aox.run_dict0
    parameters["network_data"]["n_id"] = n_id

    fig1, fig2 = azure_mockup.run_dss_simulation(parameters)
    resultdict = {
        "parameters": parameters,
        "filename": file_name,
        "plot1": base64.b64encode(fig1.getvalue()).decode("utf-8"),
        "plot2": base64.b64encode(fig2.getvalue()).decode("utf-8"),
    }

    return resultdict


# class Files(BaseModel):

#     file: str

# class SimulateParams(BaseModel):

#     files: List[Files]


# @app.get("/simulate-body")
# async def simulate_body(params: SimulateParams):

#     return "hi"
