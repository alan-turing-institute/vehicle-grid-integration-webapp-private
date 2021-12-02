import logging
from typing import Optional, List, Union
from fastapi import HTTPException
from fastapi import UploadFile
from pathlib import Path
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator
from pydantic.error_wrappers import ValidationError
from vgi_api import validation
from vgi_api.validation import (
    NetworkID,
    DefaultLV,
    VALID_LV_NETWORKS_RURAL,
    VALID_LV_NETWORKS_URBAN,
    DEFAULT_LV_NETWORKS,
)
from vgi_api.validation.types import MVEVChargerOptions, MVSolarPVOptions, ProfileUnits


class ValidateLVParams(BaseModel):

    n_id: NetworkID
    lv_list: Optional[str]
    lv_default: Optional[DefaultLV]

    @validator(
        "lv_list",
    )
    def validate_lv_list(cls, v: str, values):

        # lv_list is not required
        if v is None:
            return v

        if v == "":
            raise ValueError("No values passed")

        lv_list_int = ValidateLVParams._parse_lv_list(v)
        lv_list_len = len(lv_list_int)

        if (lv_list_len > 5) or (lv_list_len < 1):
            raise ValueError("lv_list` must be at least 1 and up to 5 items")

        valid = False
        lv_set = set(lv_list_int)
        if values["n_id"] == NetworkID.URBAN:
            urban_set = set(VALID_LV_NETWORKS_URBAN)
            valid = lv_set.issubset(urban_set)
            difference = lv_set.difference(urban_set)
        elif values["n_id"] == NetworkID.RURAL:
            rural_set = set(VALID_LV_NETWORKS_RURAL)
            valid = lv_set.issubset(rural_set)
            difference = lv_set.difference(rural_set)
        if not valid:
            raise ValueError(f"lv_list values: {list(difference)} are not network ids")

        return v

    @validator("lv_default")
    def validate_lv_default(cls, v: Optional[DefaultLV], values):

        if (v is None) and (values.get("lv_list", None) is None):
            raise ValueError("One of lv_list or lv_default must be provided")

        return v

    @classmethod
    def _parse_lv_list(cls, input: str) -> List[int]:

        return [int(i) for i in input.strip().split(",")]

    def _get_default_list(self) -> List[int]:

        return DEFAULT_LV_NETWORKS[self.n_id][self.lv_default]

    def value(self):

        if self.lv_default:
            return self._get_default_list()
        if self.lv_list:
            return


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

    try:
        params = ValidateLVParams(n_id=n_id, lv_list=lv_list, lv_default=lv_default)
    except ValidationError as e:
        raise RequestValidationError(errors=e.raw_errors)
    return params.value


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
