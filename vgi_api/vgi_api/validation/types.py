"""Types used for validating API query parameters"""
from enum import Enum
from typing import List, Dict
from pathlib import Path


class NetworkID(str, Enum):
    # See https://github.com/alan-turing-institute/e4Future-collab/blob/e1b9430c594e33736c924eaf0793ef484bd146d7/mtd-dss-digital-twins/slesNtwk_turing.py#L212
    URBAN = "1060"
    RURAL = "1061"


class AllOptions(str, Enum):

    MVSolarPVOptions = "mv-solar-pv"
    MVEVChargerOptions = "mv-ev-charger"
    LVSmartMeterOptions = "lv-smartmeter"
    LVElectricVehicleOptions = "lv-ev"
    LVPVOptions = "lv-pv"
    LVHPOptions = "lv-hp"


class DefaultLV(str, Enum):

    NEAR_SUB = "near-sub"
    NEAR_EDGE = "near-edge"
    MIXED = "mixed"


class MVSolarPVOptions(str, Enum):

    NONE = "None"
    OPTION1 = "Option a"
    CSV = "csv"


class MVEVChargerOptions(str, Enum):

    NONE = "None"
    OPTION1 = "Option a"
    CSV = "csv"


class LVSmartMeterOptions(str, Enum):
    NONE = "None"
    OPTION1 = "Option a"
    CSV = "csv"


class LVElectricVehicleOptions(str, Enum):
    NONE = "None"
    OPTION1 = "Option a"
    CSV = "csv"


class LVPVOptions(str, Enum):
    NONE = "None"
    OPTION1 = "a"
    CSV = "csv"


class LVHPOptions(str, Enum):
    NONE = "None"
    OPTION1 = "a"
    CSV = "csv"


class ProfileUnits(str, Enum):

    KW = "kW"
    KWH = "kWh"


# ToDo: Matt to give us full list of options
VALID_LV_NETWORKS_URBAN: List[int] = [1101, 1102, 1103, 1104, 1105, 1106, 1107]
VALID_LV_NETWORKS_RURAL: List[int] = []

# These are defaults that could be selected
DEFAULT_LV_NETWORKS: Dict[NetworkID, Dict[DefaultLV, List[int]]] = {
    NetworkID.RURAL: {
        DefaultLV.NEAR_SUB: [1102, 1154, 1262, 1206, 1202],
        DefaultLV.NEAR_EDGE: [1321, 1254, 1387, 1194, 1109],
        DefaultLV.MIXED: [1101, 1450, 1152, 1200, 1122],
    },
    NetworkID.URBAN: {
        DefaultLV.NEAR_SUB: [1101, 1137, 1110, 1116, 1117],
        DefaultLV.NEAR_EDGE: [1103, 1109, 1166, 1145, 1131],
        DefaultLV.MIXED: [1108, 1109, 1151, 1158, 1175],
    },
}


DATA_FOLDER = Path(__file__).parent.parent / "data"

# ToDo: Get all profiles, stick in data folder and put file name here
SOLAR_PROFILES: Dict[MVSolarPVOptions, Path] = {
    MVSolarPVOptions.OPTION1: DATA_FOLDER / "example_profile.csv",
}

EV_PROFILES: Dict[MVEVChargerOptions, Path] = {
    MVEVChargerOptions.OPTION1: DATA_FOLDER / "example_profile.csv",
}

SMART_METER_PROFILES: Dict[LVSmartMeterOptions, Path] = {
    LVSmartMeterOptions.OPTION1: DATA_FOLDER / "example_profile.csv",
}

LV_EV_PROFILES: Dict[LVElectricVehicleOptions, Path] = {
    LVElectricVehicleOptions.OPTION1: DATA_FOLDER / "example_profile.csv",
}

LV_PV_PROFILES: Dict[LVPVOptions, Path] = {
    LVPVOptions.OPTION1: DATA_FOLDER / "example_profile.csv",
}


LV_HP_PROFILES: Dict[LVHPOptions, Path] = {
    LVHPOptions.OPTION1: DATA_FOLDER / "example_profile.csv",
}
