import base64
import logging
from typing import Optional, List, Any
import copy
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
    MVFCSOptions,
    MVSolarPVOptions,
    NetworkID,
    ProfileUnits,
    response_models,
    validate_lv_parameters,
    validate_profile,
)
from vgi_api.validation.types import DEFAULT_LV_NETWORKS, AllOptions
from vgi_api.validation.validators import ValidateLVParams

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
        0.013,
        ge=0.01,
        le=0.05,
        description="Change the bandwidth (in % pu) of the oltc",
    ),
    rs_pen: float = Query(
        0.8,
        ge=0,
        le=1,
        description="Percentage residential loads",
    ),
    lv_list: Optional[str] = Query(
        None,
        description="Provide a comma seperated list of up to 5 Low Voltage Network ids. If not provided you must select an option from `lv_default`",
        example="1101, 1105, 1103",
    ),
    lv_plot_list: Optional[str] = Query(
        None,
        description="Provide a comma seperated list of up to 2 Low Voltage Network ids to plot. They must be either in `lv_list` or the networks in the `default_lv` selection",
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
    mv_fcs_profile: MVFCSOptions = Query(
        MVFCSOptions.NONE,
        description="Select a example ev profile or select CSV to upload your own. If CSV selected you must provide `mv_fcs_charger_csv`",
    ),
    mv_fcs_csv: Optional[UploadFile] = File(
        None, description="11kV connected EV fast chargers' stations"
    ),
    mv_fcs_profile_units: ProfileUnits = Query(
        ProfileUnits.KW,
        description="If `mv_fcs_charger` provided gives the units",
    ),
    lv_smart_meter_profile: LVSmartMeterOptions = Query(
        LVSmartMeterOptions.OPTION1,
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
    lv_list_validated = validate_lv_parameters(lv_list, lv_default, lv_plot_list, n_id)

    # Validate Demand and Generation Profiles
    mv_solar_profile_array = validate_profile(
        mv_solar_pv_profile, mv_solar_pv_csv, mv_solar_pv_profile_units
    )

    mv_fcs_profile_array = validate_profile(
        mv_fcs_profile, mv_fcs_csv, mv_fcs_profile_units
    )

    smart_meter_profile_array = validate_profile(
        lv_smart_meter_profile, lv_smart_meter_csv, lv_smart_meter_profile_units
    )

    lv_ev_profile_array = validate_profile(
        lv_ev_profile, lv_ev_csv, lv_ev_profile_units
    )

    lv_pv_profile_array = validate_profile(
        lv_pv_profile, lv_pv_csv, lv_pv_profile_units
    )
    lv_hp_profile_array = validate_profile(
        lv_hp_profile, lv_hp_csv, lv_hp_profile_units
    )

    logging.info("Passing params to dss")
    file_name = None

    if dry_run:
        return "valid"

    lv_plot_list = (
        ValidateLVParams._parse_lv_list(lv_plot_list)
        if lv_plot_list
        else lv_list_validated[:2]
    )

    # Pass parameters to dss
    parameters = copy.deepcopy(aox.run_dict0)

    parameters["network_data"]["n_id"] = int(n_id.value)
    parameters["network_data"]["xfmr_scale"] = xfmr_scale
    parameters["network_data"]["oltc_setpoint"] = oltc_setpoint * 100
    parameters["network_data"]["oltc_bandwidth"] = oltc_bandwidth * 100
    parameters["network_data"]["lv_sel"] = "lv_list"
    parameters["network_data"]["lv_list"] = [str(i) for i in lv_list_validated]
    parameters["rs_pen"] = rs_pen * 100
    parameters["slr_pen"] = lv_pv_pen * 100
    parameters["ev_pen"] = lv_ev_pen * 100
    parameters["hps_pen"] = lv_hp_pen * 100

    parameters["plot_options"]["lv_voltages"] = [str(i) for i in lv_plot_list]
    # Add profiles to parameters
    # ToDo: Make sure all csv uploads are in kw
    parameters["simulation_data"]["mv_solar_profile_array"] = mv_solar_profile_array
    # parameters["simulation_data"]["mv_fcs_profile_array"] = mv_ev_profile_array
    parameters["simulation_data"]["mv_fcs_profile_array"] = mv_fcs_profile_array
    parameters["simulation_data"][
        "smart_meter_profile_array"
    ] = smart_meter_profile_array
    parameters["simulation_data"]["lv_ev_profile_array"] = lv_ev_profile_array
    parameters["simulation_data"]["lv_pv_profile_array"] = lv_pv_profile_array
    parameters["simulation_data"]["lv_hp_profile_array"] = lv_hp_profile_array

    # Run simulation
    (
        mv_highlevel_buffer,
        lv_voltages_buffer,
        lv_comparison_buffer,
        mv_voltages_buffer,
        mv_powers_buffer,
        mv_highlevel_buffer,
        mv_highlevel_clean_buffer,
        trn_powers_buffer,
        profile_options_buffer,
        pmry_loadings_buffer,
        pmry_powers_buffer,
    ) = azure_mockup.run_dss_simulation(parameters)

    parameters.pop("simulation_data")
    resultdict = {
        "parameters": parameters,
        "mv_highlevel": base64.b64encode(mv_highlevel_buffer.getvalue()).decode(
            "utf-8"
        ),
        "lv_voltages": base64.b64encode(lv_voltages_buffer.getvalue()).decode("utf-8"),
        "lv_comparison": base64.b64encode(lv_comparison_buffer.getvalue()).decode(
            "utf-8"
        ),
        "mv_voltages": base64.b64encode(mv_voltages_buffer.getvalue()).decode("utf-8"),
        "mv_powers": base64.b64encode(mv_powers_buffer.getvalue()).decode("utf-8"),
        "mv_highlevel": base64.b64encode(mv_highlevel_buffer.getvalue()).decode(
            "utf-8"
        ),
        "mv_highlevel_clean": base64.b64encode(
            mv_highlevel_clean_buffer.getvalue()
        ).decode("utf-8"),
        "trn_powers": base64.b64encode(trn_powers_buffer.getvalue()).decode("utf-8"),
        "profile_options": base64.b64encode(profile_options_buffer.getvalue()).decode(
            "utf-8"
        ),
        "pmry_loadings": base64.b64encode(pmry_loadings_buffer.getvalue()).decode(
            "utf-8"
        ),
        "pmry_powers": base64.b64encode(pmry_powers_buffer.getvalue()).decode("utf-8"),
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


@app.get("/get-options", response_model=List[str])
async def get_options(option_type: AllOptions):
    def get_members(option: Any) -> List[str]:

        return [member.value for _, member in option.__members__.items()]

    if option_type == AllOptions.MVSolarPVOptions:

        return get_members(MVSolarPVOptions)

    elif option_type == AllOptions.MVFCSOptions:

        return get_members(MVFCSOptions)

    elif option_type == AllOptions.LVSmartMeterOptions:

        return get_members(LVSmartMeterOptions)

    elif option_type == AllOptions.LVElectricVehicleOptions:

        return get_members(LVElectricVehicleOptions)

    elif option_type == AllOptions.LVPVOptions:

        return get_members(LVPVOptions)

    elif option_type == AllOptions.LVHPOptions:

        return get_members(LVHPOptions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
