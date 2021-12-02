"""Types used for validating API query parameters"""
from enum import Enum
from typing import List, Dict


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
