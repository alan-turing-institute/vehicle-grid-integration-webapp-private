import logging
from typing import IO, Optional, List, Union
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
from vgi_api.validation import types
from vgi_api.validation.types import (
    MVEVChargerOptions,
    MVSolarPVOptions,
    LVSmartMeterOptions,
    LVElectricVehicleOptions,
    ProfileUnits,
    SOLAR_PROFILES,
    EV_PROFILES,
    SMART_METER_PROFILES,
    LV_EV_PROFILES,
    LV_PV_PROFILES,
    LV_HP_PROFILES,
    LVHPOptions,
    LVPVOptions,
    DATA_FOLDER,
)
import tempfile
import numpy as np
import datetime


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


class ProfileBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


def csv_to_array(file: Union[tempfile.SpooledTemporaryFile, Path]) -> np.array:
    """Convert a csv file to numpy array"""

    if isinstance(file, tempfile.SpooledTemporaryFile):
        # Make sure we're at start of file
        file.seek(0)

        ncols = len(file.readline().decode().split(","))
        return np.loadtxt(
            file, dtype=float, skiprows=0, usecols=range(1, ncols), delimiter=","
        )

    elif isinstance(file, Path):

        with file.open() as f:
            ncols = len(f.readline().split(","))

        return np.loadtxt(
            file, dtype=float, skiprows=1, usecols=range(1, ncols), delimiter=","
        )

    else:
        raise NotImplementedError(f"Not implemented for type {type(file)}")


def validate_csv(v: IO):
    """Validate an uploaded csv

    Args:
        v (IO): CSV file uploaded

    Raises:
        ValueError: Must be 48 rows of data excluding header
        ValueError: Time deltas must be in 30 minute intervals
        ValueError: All non header element in column 1 onwards must be parsable as floats

    Returns:
        IO: Return v
    """
    # Check we have the right number of lines
    expected_n_lines_excluding_header = 24 * 2
    expect_n_lines = expected_n_lines_excluding_header + 1

    lines = [l.decode().replace(" ", "").split(",") for l in v.readlines()]
    lines = [[elem.replace("\n", "") for elem in line] for line in lines]

    n_lines = len(lines)
    if n_lines != expect_n_lines:
        raise ValueError(
            f"File has {n_lines} rows. Expecting {expect_n_lines} including header row"
        )

    # Check time delta column. Every time delta should be 30 min appart
    time_deltas = []
    for r, row in enumerate(lines[1:]):

        time = datetime.datetime.strptime(row[0], "%H:%M:%S")
        delta = datetime.timedelta(
            hours=time.hour, minutes=time.minute, seconds=time.second
        )

        # Check it is 30 min after the last time
        if len(time_deltas) > 1 and (
            (delta - time_deltas[-1])
            != datetime.timedelta(hours=0, minutes=30, seconds=0)
        ):
            raise ValueError(
                f"Time on row {r+2} of file: '{delta}' is not 30 min after last row: {time_deltas[-1]}"
            )

        time_deltas.append(delta)

        # Check everything else can be a float
        for c, elem in enumerate(row[1:]):
            try:
                float(elem)
            except ValueError:
                raise ValueError(
                    f"Value on row: {r+2}, col: {c + 2} of file (value = '{elem}') cannot be parsed as float"
                )

    # If we got this far everything looks good
    return v


class MVSolarProfile(ProfileBaseModel):
    mv_solar_pv_csv: tempfile.SpooledTemporaryFile

    _validate_csv = validator("mv_solar_pv_csv", allow_reuse=True)(validate_csv)

    def to_array(self) -> np.array:

        return csv_to_array(self.mv_solar_pv_csv)


class MVEVProfile(ProfileBaseModel):
    mv_ev_charger_csv: tempfile.SpooledTemporaryFile

    _validate_csv = validator("mv_ev_charger_csv", allow_reuse=True)(validate_csv)

    def to_array(self) -> np.array:

        return csv_to_array(self.mv_ev_charger_csv)


class LVSmartMeterProfile(ProfileBaseModel):

    lv_smart_meter_csv: Optional[tempfile.SpooledTemporaryFile]

    _validate_csv = validator("lv_smart_meter_csv", allow_reuse=True)(validate_csv)

    def to_array(self) -> np.array:

        return csv_to_array(self.lv_smart_meter_csv)


class LVEVProfile(ProfileBaseModel):

    lv_ev_csv: Optional[tempfile.SpooledTemporaryFile]

    _validate_csv = validator("lv_ev_csv", allow_reuse=True)(validate_csv)

    def to_array(self) -> np.array:

        return csv_to_array(self.lv_ev_csv)


class LVPVProfile(ProfileBaseModel):

    lv_pv_csv: Optional[tempfile.SpooledTemporaryFile]

    _validate_csv = validator("lv_pv_csv", allow_reuse=True)(validate_csv)

    def to_array(self) -> np.array:

        return csv_to_array(self.lv_pv_csv)


class LVHPProfile(ProfileBaseModel):

    lv_hp_csv: Optional[tempfile.SpooledTemporaryFile]

    _validate_csv = validator("lv_hp_csv", allow_reuse=True)(validate_csv)

    def to_array(self) -> np.array:

        return csv_to_array(self.lv_hp_csv)


# class MVEVChargerProfile(BaseModel):
#     mv_ev_charger_csv: Optional[IO]


def validate_profile(
    options: Union[
        MVSolarPVOptions,
        MVEVChargerOptions,
        LVSmartMeterOptions,
        LVElectricVehicleOptions,
        LVPVOptions,
        LVHPOptions,
    ],
    csv_file: Optional[UploadFile],
) -> Optional[Path]:
    """Pass an enum of profile options: `options`. If the options enum variant is `NONE`
    will return None.

    If the variant is `CSV` it will validate the csv profiles, safe the disk and return
    a numpy array of the data. Raise an HTTP exception if no CSV is uploaded.

    If the variant is anything else it will load a profile and return it

    Args:
        options (Union[MVSolarPVOptions, MVEVChargerOptions]): A profile option
        csv_file (Optional[UploadFile]): An optional csv file. Only used if options is set to CSV

    Returns:
        Optional[np.array]: A 2D numpy array with 48 rows (30 min intervals). Each column is a profile
    """

    # ToDO: Implement
    # 1. Process uploaded CSV. Done
    # 2. Load an existing CSV. Done
    # 3. Do for all other CSV.

    if isinstance(options, MVSolarPVOptions):
        if options == MVSolarPVOptions.CSV:
            try:
                profile = MVSolarProfile(mv_solar_pv_csv=csv_file.file)
                return profile.to_array()
            except ValidationError as e:
                raise RequestValidationError(errors=e.raw_errors)
        elif options == MVSolarPVOptions.NONE:
            return None
        else:
            return csv_to_array(SOLAR_PROFILES[options])

    elif isinstance(options, MVEVChargerOptions):

        if options == MVEVChargerOptions.CSV:
            try:
                profile = MVEVProfile(mv_ev_charger_csv=csv_file.file)
                return profile.to_array()
            except ValidationError as e:
                raise RequestValidationError(errors=e.raw_errors)
        elif options == MVEVChargerOptions.NONE:
            return None
        else:
            return csv_to_array(EV_PROFILES[options])

    elif isinstance(options, LVSmartMeterOptions):

        if options == LVSmartMeterOptions.CSV:
            try:
                profile = LVSmartMeterProfile(lv_smart_meter_csv=csv_file.file)
                return profile.to_array()
            except ValidationError as e:
                raise RequestValidationError(errors=e.raw_errors)
        elif options == LVSmartMeterOptions.NONE:
            return None
        else:
            return csv_to_array(SMART_METER_PROFILES[options])

    elif isinstance(options, LVElectricVehicleOptions):

        if options == LVElectricVehicleOptions.CSV:
            try:
                profile = LVEVProfile(lv_ev_csv=csv_file.file)
                return profile.to_array()
            except ValidationError as e:
                raise RequestValidationError(errors=e.raw_errors)
        elif options == LVElectricVehicleOptions.NONE:
            return None
        else:
            return csv_to_array(LV_EV_PROFILES[options])

    elif isinstance(options, LVPVOptions):

        if options == LVPVOptions.CSV:
            try:
                profile = LVPVProfile(lv_pv_csv=csv_file.file)
                return profile.to_array()
            except ValidationError as e:
                raise RequestValidationError(errors=e.raw_errors)
        elif options == LVPVOptions.NONE:
            return None
        else:
            return csv_to_array(LV_PV_PROFILES[options])

    elif isinstance(options, LVHPOptions):

        if options == LVHPOptions.CSV:
            try:
                profile = LVHPProfile(lv_hp_csv=csv_file.file)
                return profile.to_array()
            except ValidationError as e:
                raise RequestValidationError(errors=e.raw_errors)
        elif options == LVHPOptions.NONE:
            return None
        else:
            return csv_to_array(LV_HP_PROFILES[options])

    print(type(options))
    raise HTTPException(status_code=422, detail="Option not implemented")
