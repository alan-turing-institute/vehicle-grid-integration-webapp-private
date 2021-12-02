from typing import Optional, List, Union
from fastapi import HTTPException
from fastapi import UploadFile
from pathlib import Path
from vgi_api.validation import (
    NetworkID,
    DefaultLV,
    VALID_LV_NETWORKS_RURAL,
    VALID_LV_NETWORKS_URBAN,
    DEFAULT_LV_NETWORKS,
)
from vgi_api.validation.types import MVEVChargerOptions, MVSolarPVOptions, ProfileUnits


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
        List[int]: A list of network ids
    """

    def _validate_lv_list(
        lv_list: Optional[str], n_id: NetworkID
    ) -> Optional[List[int]]:

        if not lv_list:
            return

        validation_error = HTTPException(
            status_code=422, detail="lv_list values are not valid"
        )

        lv_list_int = [int(i) for i in lv_list.strip().split(",")]

        if len(lv_list_int) > 5:
            raise HTTPException(
                status_code=422, detail="lv_list` must be 5 items or less"
            )

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


def validate_profile(
    options: Union[MVSolarPVOptions, MVEVChargerOptions],
    csv_file: Optional[UploadFile],
    csv_profile_units: ProfileUnits,
) -> Optional[Path]:
    """Pass an enum of profile options: `options`. If the options enum variant is `NONE`
    will return None.

    If the variant is `CSV` it will validate the csv profiles, safe the disk and return
    the absolute path to the csv.

    If the variant is anything else it will return the absolute path to a pre-existing
    csv profile

    Args:
        options (Union[MVSolarPVOptions, MVEVChargerOptions]): A profile option
        csv_file (Optional[UploadFile]): An optional csv file. Only used if options is set to CSV
        csv_profile_units (ProfileUnits): The units of the CSV file.

    Returns:
        Optional[Path]: A Path to a csv profile on disk
    """

    # ToDO: Implement
    return None
