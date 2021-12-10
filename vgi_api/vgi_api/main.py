import base64
import logging
from typing import Optional

import fastapi
from fastapi import File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from vgi_api import azure_mockup
from vgi_api import azureOptsXmpls as aox
from vgi_api.validation import (
    VALID_LV_NETWORKS_RURAL,
    VALID_LV_NETWORKS_URBAN,
    DefaultLV,
    LVElectricVehicleOptions,
    LVHPOptions,
    LVPVOptions,
    LVSmartMeterOptions,
    MVEVChargerOptions,
    MVSolarPVOptions,
    NetworkID,
    ProfileUnits,
    response_models,
    validate_lv_parameters,
    validate_profile,
)
from vgi_api.validation.types import DEFAULT_LV_NETWORKS

app = fastapi.FastAPI()

origins = ["http://localhost:8080", "http://192.168.1.63:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/simulate")
async def simulate(
    dry_run: bool = Query(
        False,
        title="Dry run",
        description="Check all simulation arguments are valid without running simulation",
    ),
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
        1.04,
        ge=0.95,
        le=1.10,
        description="Medium Voltage transformer on-load tap charger (OLTC) set point. Change the set point (in % pu) of the oltc",
    ),
    oltc_bandwidth: float = Query(
        0.13,
        ge=0.1,
        le=0.5,
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
        description="Provide a comma seperated list of up to 5 Low Voltage Network ids. If not provided you must select an option from `lv_default`",
        example="1101, 1105, 1103",
    ),
    lv_default: Optional[DefaultLV] = Query(
        None,
        description="Choose a default set of Low Voltage Networks. If `lv_list` is provided `lv_list` takes precedence",
    ),
    mv_solar_pv_profile: MVSolarPVOptions = Query(
        MVSolarPVOptions.NONE,
        description="Select a example solar pv profile or select CSV to upload your own. If CSV selected you must provide `mv_solar_pv_csv`",
    ),
    mv_solar_pv_csv: Optional[UploadFile] = File(
        None, description="11kV connected solar photovoltaic (PV)"
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
        None, description="11kV connected EV fast chargers' stations"
    ),
    mv_ev_charger_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="If `mv_ev_charger` provided gives the units",
    ),
    lv_smart_meter_profile: LVSmartMeterOptions = Query(
        LVSmartMeterOptions.NONE,
        description="",
    ),
    lv_smart_meter_csv: Optional[UploadFile] = File(None, description=""),
    lv_smart_meter_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_ev_profile: LVElectricVehicleOptions = Query(
        LVElectricVehicleOptions.NONE,
        description="",
    ),
    lv_ev_csv: Optional[UploadFile] = File(None, title=""),
    lv_ev_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_ev_pen: float = Query(
        0.0,
        ge=0,
        le=1,
        description="Percentage Electric Vehicle Penetration",
    ),
    lv_pv_profile: LVPVOptions = Query(
        LVPVOptions.NONE,
        description="",
    ),
    lv_pv_csv: Optional[UploadFile] = File(None, title=""),
    lv_pv_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_pv_pen: float = Query(
        0.0,
        ge=0,
        le=1,
        description="Percentage PV Penetration",
    ),
    lv_hp_profile: LVHPOptions = Query(
        LVHPOptions.NONE,
        description="",
    ),
    lv_hp_csv: Optional[UploadFile] = File(None, title=""),
    lv_hp_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="",
    ),
    lv_hp_pen: float = Query(
        0.0,
        ge=0,
        le=1,
        description="Percentage Heat Pump Penetration",
    ),
):

    # MV parameters are already valid. LV parameters need additional validation
    lv_list_validated = validate_lv_parameters(lv_list, lv_default, n_id)

    # Validate Demand and Generation Profiles
    mv_solar_profile_array = validate_profile(mv_solar_pv_profile, mv_solar_pv_csv)

    ## ToDo: Add validation for all other files types
    mv_ev_profile_array = validate_profile(mv_ev_charger_profile, mv_ev_charger_csv)

    smart_meter_profile_array = validate_profile(
        lv_smart_meter_profile, lv_smart_meter_csv
    )

    lv_ev_profile_array = validate_profile(lv_ev_profile, lv_ev_csv)

    lv_pv_profile_array = validate_profile(lv_pv_profile, lv_pv_csv)
    lv_hp_profile_array = validate_profile(lv_hp_profile, lv_hp_csv)

    logging.info("Passing params to dss")
    file_name = None

    if dry_run:
        return "valid"

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


@app.get("/lv-network", response_model=response_models.LVNetworks)
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


@app.get("/lv-network-defaults", response_model=response_models.LVNetworks)
async def lv_network_defaults(
    n_id: NetworkID = Query(
        ...,
        title="Network ID",
        description="Choice of 11 kV integrated MV-LV network",
    ),
    lv_default: DefaultLV = Query(
        ...,
        description="Choose a default set of Low Voltage Networks. If `lv_list` is provided `lv_list` takes precedence",
    ),
):
    """Return the network ids of the network option"""

    return {"networks": DEFAULT_LV_NETWORKS[n_id][lv_default]}


# def save_uploaded_file(file, save_name, size_limit):
#     real_file_size = 0
#     temp = NamedTemporaryFile(delete=False)
#     for chunk in file.file:
#         real_file_size += len(
#             chunk
#         )  # Chunk size (default 1MB) set by starlette https://github.com/encode/starlette/blob/master/starlette/datastructures.py#L412
#         if real_file_size > size_limit:
#             logging.info(
#                 "File has reached size {} > limit {}".format(real_file_size, size_limit)
#             )
#             raise HTTPException(
#                 status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
#                 detail="{} is too large".format(file.filename),
#             )
#         temp.write(chunk)
#     temp.close()
#     shutil.move(temp.name, save_name)
