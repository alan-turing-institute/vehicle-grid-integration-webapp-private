"""Types used for validating API query parameters"""
from enum import Enum
from typing import List, Dict


class NetworkID(str, Enum):
    # See https://github.com/alan-turing-institute/e4Future-collab/blob/e1b9430c594e33736c924eaf0793ef484bd146d7/mtd-dss-digital-twins/slesNtwk_turing.py#L212
    URBAN = "1060"
    RURAL = "1061"


class DefaultLV(str, Enum):

    NEAR_SUB = "near-sub"
    NEAR_EDGE = "near-edge"
    MIXED = "mixed"


class MVSolarPVOptions(str, Enum):

    NONE = "None"
    OPTION1 = "Option a"
    OPTION2 = "Option b"
    OPTION3 = "Option c"
    CSV = "csv"


class MVEVChargerOptions(str, Enum):

    NONE = "None"
    OPTION1 = "Option d"
    OPTION2 = "Option e"
    OPTION3 = "Option f"
    CSV = "csv"


class LVSmartMeterOptions(str, Enum):
    NONE = "None"
    OPTION1 = "Option g"
    OPTION2 = "Option h"
    OPTION3 = "Option i"
    CSV = "csv"


class LVElectricVehicleOptions(str, Enum):
    NONE = "None"
    OPTION1 = "Option j"
    OPTION2 = "Option k"
    OPTION3 = "Option l"
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


# lv_example_options_all = {
# Urban
#     1060:{

#     'near_sub':['1101','1137','1110','1116','1117',],

#     'near_edge':['1103','1109','1166','1145','1131',],

#     'mixed':['1108','1109','1151','1158','1175',],

#     },

#     1061:{

#     'near_sub':['1102','1154','1262','1206','1202',],

#     'near_edge':['1321','1254','1387','1194','1109',],

#     'mixed':['1101','1450','1152','1200','1122',],

#     },

# }
