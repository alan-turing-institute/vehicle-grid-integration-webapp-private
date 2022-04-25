# =====
# SCRIPT AUTOMATICALLY GENERATED!
# =====
#
# This means it is liable to being overwritten, so it is suggested NOT
# to change this script, instead make a copy and work from there.
#
# Code for generating these in ./misc/copy-mtd-funcs-raw.py.
#
# =====

import glob
import os, sys, pickle, timeit, copy, time
import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib import rcParams
import matplotlib.cm as cm
from scipy import sparse
from os.path import join
from itertools import chain
import matplotlib.cm as mplcm
from copy import deepcopy
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from pprint import pprint
from collections import OrderedDict as odict

from .funcsMath_turing import (
    aMulBsp,
    tp2ar,
    vecSlc,
    pf2kq,
    vmM,
    mvM,
    schurSolve,
    calcDp1rSqrt,
    rerr,
    sparseInvUpdate,
    dp1rMv,
    calcVar,
    magicSlice,
    sparseBlkDiag,
    tp2arL,
)
from .funcsDss_turing import dssIfc, updateFlagState, pdIdxCmp, getVset, phs2seq
from .funcsPython_turing import (
    fPrmsInit,
    whocalledme,
    saveFigFunc,
    sff,
    struct,
    mtList,
    mtDict,
    cleanFolder,
    structDict,
    fn_ntwx,
    fn_root,
    o2o,
    rngSeed,
    sff_legacy,
    cmsq,
)
import dss

from importlib import reload

d = dssIfc(dss.DSS)
# Preamble END ---


class dNet:
    """Python-OpenDSS class with networks built-in."""

    fdrs = {
        5: "13bus",
        6: "34bus",
        7: "37bus",
        8: "123bus",
        9: "8500node",
        14: "usLv",
        17: "epri5",
        18: "epri7",
        19: "epriJ1",
        20: "epriK1",
        21: "epriM1",
        22: "epri24",
        23: "4busYy",
        24: "epriK1cvr",
        25: "epri24cvr",
        26: "123busCvr",
        27: "33bus",
        30: "eulv",
        31: "4busLv2lv",
        32: "4busLv2fdr",
        33: "33bus-eulv",
        34: "33bus-10lv",
        35: "33bus-33lv",
        36: "33bus-10lv_p",
        37: "33bus-33lv_p",
        # 38:'HV_UG_full_cut',
        38: "HV_OHa_full_cut",
        39: "HV_UG_full_one",
        # 40:'turing_working',
        40: "HV_UG-OHa_full_small",
        50: "HV_UG",
        51: "HV_UG-OHa",
        52: "HV_UG-OHb",
        53: "HV_OHa",
        54: "HV_OHb",
        55: "HV_OH-UGa",
        56: "HV_OH-UGb",
        60: "HV_UG_full",
        61: "HV_UG-OHa_full",
        62: "HV_UG-OHb_full",
        63: "HV_OHa_full",
        64: "HV_OHb_full",
        65: "HV_OH-UGa_full",
        66: "HV_OH-UGb_full",
        70: "HV_UG_mod",
        71: "HV_UG-OHa_mod",
        72: "HV_UG-OHb_mod",
        73: "HV_OHa_mod",
        74: "HV_OHb_mod",
        75: "HV_OH-UGa_mod",
        76: "HV_OH-UGb_mod",
        80: "HV_UG_full_lo",
        81: "HV_UG-OHa_full_lo",
        82: "HV_UG-OHb_full_lo",
        83: "HV_OHa_full_lo",
        84: "HV_OHb_full_lo",
        85: "HV_OH-UGa_full_lo",
        86: "HV_OH-UGb_full_lo",
        90: "HV_UG_full_hi",
        91: "HV_UG-OHa_full_hi",
        92: "HV_UG-OHb_full_hi",
        93: "HV_OHa_full_hi",
        94: "HV_OHb_full_hi",
        95: "HV_OH-UGa_full_hi",
        96: "HV_OH-UGb_full_hi",
        126: "n7f1",
        127: "n1cvr",
        128: "n1_33bus_ld31",
        "101-125": "Manc. LVNS, as-is",
        "201-225": "Manc. LVNS, pruned version",
        "226-250": "Manc. LVNS, pruned + UKGDS edits",
        "251-275": "Manc. LVNS, pruned + UKGDS, xfmr Z modded",
        "276-300": "Manc. LVNS, pruned + UKGDS fdr one, xfmr Z modded",
        "301-330": "Leuven LV networks",
        1000: "_network_mod",
        1060: "HV_UG_full_turing",
        1061: "HV_UG-OHa_full_turing",
        # 1062:'HV_UG-OHb_full',
        # 1063:'HV_OHa_full',
        # 1064:'HV_OHb_full',
        # 1065:'HV_OH-UGa_full',
        # 1066:'HV_OH-UGb_full',
    }
    fdrs_inv = {v: k for k, v in fdrs.items()}

    # 'HV_UG-OHa_full_small':str(fdrs_inv['HV_UG-OHa_full_small']),
    dn2frids = copy.copy(fdrs_inv)
    dn2frids.update(
        {
            "ukgds-master-dss": "50-56 incl.",
        }
    )

    fdrsLoc = {
        5: join("13Bus_copy\\IEEE13Nodeckt_z"),
        6: join("34Bus_copy", "ieee34Mod1_z"),
        7: join("37Bus_copy", "ieee37_z"),
        8: join("123Bus_copy", "IEEE123Master_z"),
        9: join("8500-Node_copy", "Master-unbal_z"),
        14: join("usLv", "master_z"),
        17: join("ckt5", "Master_ckt5_z"),
        18: join("ckt7", "Master_ckt7_z"),
        19: join("j1", "Master_noPV_z"),
        20: join("k1", "Master_NoPV_z"),
        21: join("m1", "Master_NoPV_z"),
        22: join("ckt24", "master_ckt24_z"),
        23: join("4Bus-YY-Bal", "4Bus-YY-Bal_z"),
        24: join("k1", "Master_NoPV_z_cvr"),
        25: join("ckt24", "master_ckt24_z_cvr"),
        26: join("123Bus_copy", "IEEE123Master_cvr"),
        27: join("33bus", "ieee33master"),
        30: join("LVTestCase_copy", "master_z"),
        31: join("4busLv2lv", "4busLv2lv"),
        32: join("4busLv2fdr", "4busLv2fdr"),
        33: join("33bus-eulv", "ieee33-eulv-master"),
        34: join("33bus-33lv", "ieee33-10lv-master"),
        35: join("33bus-33lv", "ieee33-33lv-master"),
        36: join("33bus-33lv", "ieee33-10lv-master_pruned"),
        37: join("33bus-33lv", "ieee33-33lv-master_pruned"),
        # 38:join('HV_UG_full_cut','master_full_cut'),
        38: join("HV_OHa_full_cut", "master_full_cut"),
        39: join("HV_UG_full_one", "master_full_one"),
        40: join("HV_UG-OHa_full_small", "master_mvlv_small"),
        50: join("HV_UG", "master"),
        51: join("HV_UG-OHa", "master"),
        52: join("HV_UG-OHb", "master"),
        53: join("HV_OHa", "master"),
        54: join("HV_OHb", "master"),
        55: join("HV_OH-UGa", "master"),
        56: join("HV_OH-UGb", "master"),
        60: join("HV_UG_full", "master_full"),
        61: join("HV_UG-OHa_full", "master_full"),
        62: join("HV_UG-OHb_full", "master_full"),
        63: join("HV_OHa_full", "master_full"),
        64: join("HV_OHb_full", "master_full"),
        65: join("HV_OH-UGa_full", "master_full"),
        66: join("HV_OH-UGb_full", "master_full"),
        70: join("HV_UG", "master"),
        71: join("HV_UG-OHa", "master"),
        72: join("HV_UG-OHb", "master"),
        73: join("HV_OHa", "master"),
        74: join("HV_OHb", "master"),
        75: join("HV_OH-UGa", "master"),
        76: join("HV_OH-UGb", "master"),
        80: join("HV_UG_full", "master_full_lo"),
        81: join("HV_UG-OHa_full", "master_full_lo"),
        82: join("HV_UG-OHb_full", "master_full_lo"),
        83: join("HV_OHa_full", "master_full_lo"),
        84: join("HV_OHb_full", "master_full_lo"),
        85: join("HV_OH-UGa_full", "master_full_lo"),
        86: join("HV_OH-UGb_full", "master_full_lo"),
        90: join("HV_UG_full", "master_full_hi"),
        91: join("HV_UG-OHa_full", "master_full_hi"),
        92: join("HV_UG-OHb_full", "master_full_hi"),
        93: join("HV_OHa_full", "master_full_hi"),
        94: join("HV_OHb_full", "master_full_hi"),
        95: join("HV_OH-UGa_full", "master_full_hi"),
        96: join("HV_OH-UGb_full", "master_full_hi"),
        126: join("network_7", "masterNetwork7feeder1"),
        127: join("network_27", "masterNetwork1"),
        128: join("network_28", "masterNetwork1"),
        1000: join("_network_mod", "master_mvlv"),
        1060: join("HV_UG_full", "master_mvlv"),
        1061: join("HV_UG-OHa_full", "master_mvlv"),
        # 1062:join('HV_UG-OHb_full','master_full'),
        # 1063:join('HV_OHa_full','master_full'),
        # 1064:join('HV_OHb_full','master_full'),
        # 1065:join('HV_OH-UGa_full','master_full'),
        # 1066:join('HV_OH-UGb_full','master_full'),
    }

    # Initial constraint numbers
    cns = structDict(
        {
            "vMv": structDict({"U": 1.05, "L": 0.95}),
            "vLv": structDict({"U": 1.05, "L": 0.92}),
            "Dv": structDict({"U": 0.06, "L": -np.inf}),
            "Ixfm": structDict({"U": 1.50, "L": -np.inf}),
            "Vub": structDict({"U": 0.02, "L": -np.inf}),
        }
    )
    cns.N = len(cns.kw)

    # Data for the saved caches
    caches = {
        "Ylli": {
            "name": "Ylli",
            "dn_": "YlliCache",
            "fn_": "Ylli_",
            "res_": {"Ylli", "capTap"},
        },
        # 'Ybus':{
        # 'name':'Ybus',
        # 'dn_':'YbusCache',
        # 'fn_':'Ybus_',
        # 'res_':{'Ybus','Yprm','YbusState','YbusCapTap'}
        # },
    }

    # capTaps for testing funcs
    capTapTests = {
        "13bus": [[(0,), (1,)], [9, 9, 9]],
        "34bus": [[(0,)], [6, 1, 2, 1, 1, 1]],
        "epri7": [[(0,), (0,)], []],
        "epriK1": [[(0,)], [1]],
        "n1_p": [[], []],  # 201
    }

    # For basic solution testing:
    slnVals = {
        "stateIn": {},
        "VsrcMeas": np.nan,
        "Vc": np.nan,
        "vc": np.nan,
        "V": np.nan,
        "v": np.nan,
        "stateOut": {},
        "TP": np.nan,
        "TL": np.nan,
        "Cnvg": np.nan,
    }

    _d = d

    # Number of feeders per LV circuit - e.g. ckt 1 has 4x feeders.
    # From miscScripts\explore_manc_ntwx.
    fdrNos = [
        np.nan,
        4,
        5,
        6,
        6,
        8,
        2,
        7,
        2,
        6,
        6,
        5,
        3,
        4,
        6,
        7,
        4,
        7,
        9,
        5,
        5,
        5,
        6,
        5,
        2,
        3,
    ]

    # Some plotting data - fPrm_ as figure parameters
    fPrm_ = {
        "cmap": None,
        "legLoc": None,  # e.g. 'NorthEast'. Not used...?
        "tFs": 10,  # title font size
        "ttlOn": False,
        "cms": {
            "v": cm.RdBu,
            "p": cm.GnBu,
            "q": cm.PiYG,
            "n": cm.Blues,
        },
        "pms": 50,
    }

    # 3 phase plotting scale of hexagons
    sfDict3ph = {
        "13bus": 6,
        "123bus": 60,
        "123busCvr": 60,
        "epriK1cvr": 40,
        "34bus": 80,
        "n1": 4,
        "eulv": 2,
        "epri5": 90,
        "epri7": 50,
        "n10": 3,
        "n4": 3,
        "n27": 4,
        "33bus": 12,
    }

    def __init__(
        self,
        frId=30,
        yBdInit=True,
        prmIn={},
        mod_dir=os.path.join(os.getcwd(), "_network_mod"),
        **kwargs,
    ):
        """Initialise with kwargs.

        =====
        Inputs
        =====
        frId    - the feeder ID. Use self.getCktFn(1) to get a list of options
        yBdInit - if true create derived matrices
        prmIn: a dict of nominal parameter updates:
                - nomLdLm: 0.6, load mult at the initial solution
                - pCvr: 0, p cvr coefficient at initial solution
                - qCvr: 0, q cvr coefficient at initial solution
                - tolStr: the chosen opendss solution tolerance
                - yset: 'Ybus' or 'both', the latter also getting self.Yprm
        mod_dir - directory where modified OpenDSS files are stored. Used only for frID 1000.

        =====
        kwargs
        =====
        sd: save directory for figures
        tests: list of named tests,
                - 'ybus' for checking ybus matches opendss explicitly
                - 'ykvbase' for checking the voltage bases

        """

        self.mod_dir = mod_dir

        self.prm = {
            "nomLdLm": 0.6,
            "pCvr": 0,
            "qCvr": 0,
            "tolStr": "1e-6",
            # 'tolStr':'1e-10',
            "yset": "both",
        }
        self.prm.update(prmIn)

        # load in tests list
        self.tests = kwargs.get("tests", ["ybus"])

        # figure parameters
        self.fPrm = fPrmsInit(kwargs.get("sd", __file__))

        if "lm" in kwargs.keys():
            raise Exception("'lm' not supported by dNet! Use prmIn.")

        # get ckt info
        self.frId = frId
        self.setCktFn()

        # Get initial opendss state
        print("Initialise OpenDSS, Circuit, then Get Circuit Parameters.")
        self.initialiseOpenDss()
        self.initialiseDssCkt()

        self.state0 = d.getNtwkState()
        self.getIdxPhs()

        # Get parameters of the circuit
        state_ = self.state0.copy()
        state_["CtrlMode"] = -1
        d.setNtwkState(state_)

        self.nV = d.DSSCircuit.NumNodes - 3  # following previous conventions

        self.YZ = d.DSSCircuit.YNodeOrder
        self.YZidxs = {bus: i for i, bus in enumerate(self.YZ)}
        self.YZsrc = self.YZ[:3]
        self.YZv = self.YZ[3:]  # following previous conventions
        self.YZvIdxs = {bus: i for i, bus in enumerate(self.YZv)}
        self.getSourceBus()

        # Get derived quantities
        self._V = np.abs(tp2ar(d.DSSCircuit.YNodeVarray))
        self.vKvbase = d.get_Yvbase(self.YZ, test=("ykvbase" in self.tests))[3:]
        dNet.getVolLevels(self)

        # NB slightly different from prev conv.
        self.getCounts()

        self.YZsy = vecSlc(self.YZv, self.pyIdx)
        self.YZsd = vecSlc(self.YZv, self.pdIdx)
        self.YZs = vecSlc(self.YZv, self.sIdx)

        self.qyIdx = self.pyIdx + len(self.YZv)
        self.syIdx = np.r_[self.pyIdx, self.qyIdx]
        self.qdIdx = self.pdIdx + len(self.YZv)
        self.sdIdx = np.r_[self.pdIdx, self.qdIdx]

        self.pdIdxCmp, self.pdIdxUnq = pdIdxCmp(self.YZv, self.pdIdx)

        # Initiliase self.sln
        self.getSln(None)

        # NB isn't much faster than the previous buildHmat code.
        self.buildSpHmat()

        # Get the admittance matrix
        self.getYset(
            yset=self.prm["yset"], state=self.state0, testYbus=("ybus" in self.tests)
        )

        # Get derived matrices
        if yBdInit:
            self.getUbMats(test=False)

            self.getYprmBd()
            self.getYprmBd(prmType="YprmV", dssType=d.TRN, psConv=True)

            self.xfmRatings = d.getXfmrIlims(d.getRegXfms())

            self.nIprm = self.YprmBd.shape[0]
            self.Vnl = -spla.spsolve(self.Ybus[3:, 3:], self.Ybus[3:, :3]).dot(
                d.getVsrc()
            )

    def getCounts(self):
        """Get the counts and indexes of LOADS.

        Indexes are with respect to YZv, not YZ.

        No generators are assumed for now.
        """
        self.updateLdsCount()
        self.updateGenCount()

        # WARNING: assumes that Loads and Generators are at the same buses.
        self.nPy = len(self.pyIdx)
        self.nPd = len(self.pdIdx)
        self.nP = self.nPy + self.nPd
        self.nS = self.nP * 2
        self.nT = len(self.state0["capTap"][1])
        self.nC = len(self.cap2list(self.state0["capTap"][0]))

    def updateLdsCount(self):
        # Note that these are all generally different from n.LDS.Count.
        sIdx, self.LDSidx, self.LDS_ = d.getPwrIdxs(
            d.LDS,
            self.YZv,
        )
        self.pyIdx, self.pdIdx = sIdx
        self.sIdx = np.concatenate(sIdx)
        self.nPyLds = len(self.LDS_.Y.idx)
        self.nPdLds = len(self.LDS_.D.idx)

    def updateGenCount(self):
        _, self.GENidx, self.GEN_ = d.getPwrIdxs(
            d.GEN,
            self.YZv,
        )
        # self.pyIdx, self.pdIdx = sIdx
        self.nPyGen = len(self.GEN_.Y.idx)
        self.nPdGen = len(self.GEN_.D.idx)

    @staticmethod
    def cap2list(capList):
        return list(chain(*capList))

    def clearCache(
        self,
        fldr,
    ):
        """Clear a cache.

        fldr opts are from help(self.getCacheFn).
        """
        dn = self.getCacheFn(fldr, getDn=True)
        cleanFolder(dn)

    def getYset(self, yset="both", **kwargs):
        """Get the admittance matrix at the given state.

        Inputs
        ---
        yset: either 'Ybus', 'Yprm', 'both' for Ybus or Yprm matrices (or both)

        kwargs
        ---
        fixCtrl: to fix the control or not
        state: A state to get the admittance matrix at. Note that 'fixCtrl'
               changes the state then returns it.
        capTap: An alternative to setting state.
        testYbus: If true, load the admittance matrix from opendss & compare

        Returns
        ---
        Sets the following attributes (through calcYset or updateYset)
        - yset
        - yset+'State'
        - yset+'CapTap'
        - yset+'Kwargs'

        """
        # Unpack kwargs
        capTap = kwargs.get("capTap", None)
        state = kwargs.get("state", deepcopy(self.state0))
        fixCtrl = kwargs.get("fixCtrl", True)

        # in case we want to rerun later
        if yset in ["both", "Ybus"]:
            self.YbusKwargs = kwargs
        if yset in ["both", "Yprm"]:
            self.YprmKwargs = kwargs

        if not capTap is None:
            if "state" in kwargs.keys():
                raise Exception("Specify state XOR capTap, not both.")

            state["capTap"] = capTap
            state["CtrlMode"] = -1
        else:
            # Get the state
            capTap = state["capTap"]

        CtrlMode_ = state["CtrlMode"]

        if fixCtrl:
            # Force turn-off controls
            state["CtrlMode"] = -1

        sKeys = self.__dict__.keys()

        # Update without fully rebuilding where possible.
        if (yset == "both" and "Ybus" in sKeys and "Yprm" in sKeys) or yset in sKeys:
            capTap_ = getattr(self, "YbusCapTap").copy()
            self.updateYset(
                capTap_,
                capTap,
                yset,
            )
        else:
            self.calcYset(
                state,
                yset,
            )

        # Set the control mode back to initial in case it is used elsewhere
        state["CtrlMode"] = CtrlMode_

        if kwargs.get("testYbus", False):
            Ybus2 = d.createYbus(state["capTap"])[0]
            val = spla.norm(self.Ybus - Ybus2) / spla.norm(Ybus2)
            print("YBus Equal:", val < 1e-14)
            print("YBus Equal:", val)
            self.Ybus2 = Ybus2

    def updateYset(
        self,
        capTap_,
        capTap,
        yset="both",
    ):
        """Update Ybus/Yprm to the new capTap position capTap.

        Only updates if capTap_ is different from capTap.

        Inputs
        ---
        capTap_: passed as first argument to getOldNewYmats
        capTap: passed as second argument to getOldNewYmats
        yset: the yset to update ('Ybus', 'Yprm', or 'both')

        """

        if capTap != capTap_:
            (YmOld, YmNew), (YpO, YpN) = self.getOldNewYmats(capTap_, capTap)

            if yset in ["Ybus", "both"]:
                self.Ybus = self.Ybus + YmNew - YmOld
                self.Ybus.eliminate_zeros()
                self.YbusState = d.getNtwkState()
                self.YbusCapTap = self.YbusState["capTap"]

            if yset in ["Yprm", "both"]:
                self.Yprm.update(YpN)
                self.YprmState = d.getNtwkState()
                self.YprmCapTap = self.YbusState["capTap"]

        else:
            pass

    def getOldNewYmats(self, capTap_, capTap):
        """Get primitive and admittance matrix perturbations.

        capTap_ is the OLD capTap,
        capTap is the NEW capTap.

        Returns (YmatOld,YmatNew),(YprmsOld,YprmsNew).
        """
        CtrlMode_ = d.setControlMode(-1)

        # Get the OLD capTap Ymats
        YmatCap_ = d.buildYmat(d.CAP, capTap_, self.YZidxs)[:2]
        YmatTap_ = d.buildYmat(d.RGC, capTap_, self.YZidxs)[:2]

        # Get the NEW capTap Ymats
        YmatCap = d.buildYmat(d.CAP, capTap, self.YZidxs)[:2]
        YmatTap = d.buildYmat(d.RGC, capTap, self.YZidxs)[:2]

        YmatOld = YmatCap_[0] + YmatTap_[0]
        YmatNew = YmatCap[0] + YmatTap[0]

        YprmsOld = {**YmatCap_[1], **YmatTap_[1]}
        YprmsNew = {**YmatCap[1], **YmatTap[1]}

        d.setControlMode(CtrlMode_)
        return (YmatOld, YmatNew), (YprmsOld, YprmsNew)

    def calcYset(
        self,
        state,
        yset="Ybus",
    ):
        """A wrapper to pull out Ybus, Yprm, or both.

        If 'both' is called then does a simple error check on the built
        and actual Ybus matrices.

        Inputs
        ---
        state: the state to get the Ybus at
        yset: as in getYset [can be 'both']

        Sets attributes:
        ---
        - yset
        - yset+'State'
        - yset+'CapTap'

        """
        d.setNtwkState(state)

        Res = {}
        if yset in ["Ybus", "both"]:
            Res.update({"Ybus": d.getYmat(None, True)[0]})
            Res.update({"YbusState": d.getNtwkState()})
            Res.update({"YbusCapTap": Res["YbusState"]["capTap"]})
        if yset in ["Yprm", "both"]:
            Ybus2, Yprm = d.buildYmat(
                d.PDE,
                state["capTap"],
                self.YZidxs,
                1,
            )[:2]
            Res.update({"Yprm": Yprm})
            Res.update({"YprmState": d.getNtwkState()})
            Res.update({"YprmCapTap": Res["YbusState"]["capTap"]})

        # Do a quick error check if getting both
        if yset == "both":
            if rerr(Res["Ybus"], Ybus2, p=0) > 1e14:
                print("Warning! Ybus and Ybus2 do not match!")

        self.dictSetAttr(Res)
        return Res

    def dictSetAttr(self, dct):
        """A simple wrapper to set a dictionary of attributes to self."""
        for key, val in dct.items():
            setattr(self, key, val)

    def getIdxPhs(self):
        """Run through Y- and D- LDS+GEN, record each as one-or 3-phase."""
        self.iIdxPhsY = np.zeros(
            len(self.state0["LdsY"]) + len(self.state0["GenY"]), dtype=int
        )
        self.iIdxPhsD = np.zeros(
            len(self.state0["LdsD"]) + len(self.state0["GenD"]), dtype=int
        )

        iY = 0
        iD = 0
        for obj in [d.LDS, d.GEN]:
            i = obj.First
            while i:
                if d.AE.Properties("conn").Val.lower() == "delta":
                    self.iIdxPhsD[iD] = d.ACE.NumPhases
                    iD += 1
                else:
                    self.iIdxPhsY[iY] = d.ACE.NumPhases
                    iY += 1
                i = obj.Next

    def initialiseOpenDss(self):
        """Get the dssobj into self for reference by other objects."""
        self.DSSObj = d.DSSObj

    def initialiseDssCkt(self):
        """Reset OpenDSS and compile the file.

        Remember! This resets gets rid of any changes
        e.g. self.allGenSetup()
        DSSText.Command='Compile ('+self.fn+')'

        Note that this can be quite slow for the big networks.
        """
        d.resetDss()


        cur_dir = os.getcwd()
        # The next line changes the current working directory so we change it back
        d.DSSText.Command = "Compile " + self.fn
        os.chdir(cur_dir)
        d.DSSText.Command = "Set tol=" + self.prm["tolStr"]
        d.SLN.Solve()

        d.DSSText.Command = (
            "Batchedit load..* vminpu=0.02 vmaxpu=50"
            + " model=4 status=variable cvrwatts="
            + str(self.prm["pCvr"])
            + " cvrvars="
            + str(self.prm["qCvr"])
        )

        d.SLN.LoadMult = self.prm["nomLdLm"]
        d.SLN.Solve()

    def buildSpHmat(self):
        """A fast way of building a sparse instantiation of the H-matrix."""
        dataSet = np.array([np.ones(self.nPd), -np.ones(self.nPd)]).T.flatten()
        iSet = np.array([np.arange(self.nPd)] * 2).T.flatten()
        jSet = np.array([self.pdIdx, self.pdIdxCmp]).T.flatten()
        self.Hmat = sparse.coo_matrix(
            (dataSet, (iSet, jSet)), (self.nPd, self.nV)
        ).tocsr()

    def setCktFn(
        self,
    ):
        """Set self.feeder, self.fn by calling getCktFn."""
        self.feeder, self.fn = self.getCktFn()

    def getCktFn(self, printF=False, frId=None):
        """A function to get circuit feederId and fn, or print.

        Use printF to print all options;
        Use frId to return details of other circuits.

        """
        if frId is None:
            frId = self.frId

        if printF:
            pprint(self.fdrs)
        if len(str(frId)) < 3 or len(str(frId)) >= 4:
            if frId in [
                5,
                6,
                7,
                8,
                9,
                14,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
            ]:
                ND = join(fn_ntwx, "ieee_tn")
            elif frId in [30, 38, 39, 40]:
                ND = fn_ntwx
            # elif frId in [40]:
            # ND = r"D:\codeD\uk-mvlv-models-mod"
            elif frId in list(range(50, 57)):
                ND = join(fn_ntwx, "ukgds-master-dss")
            elif frId in list(range(60, 67)) + list(range(80, 87)) + list(
                range(90, 97)
            ):
                ND = join(fn_ntwx, "ukgds-master-full")
            elif frId in list(range(70, 77)):
                ND = join(fn_ntwx, "ukgds-master-mod")
            elif frId == 1000:
                ND = self.mod_dir
            elif frId > 1000:
                ND = join(
                    fn_ntwx,
                )

        elif len(str(frId)) == 3 and frId < 301:
            ND, feederId, masterFn = dNet.getLvnsFn(frId)
            self.fdrs[frId] = feederId
            self.fdrsLoc[frId] = masterFn

        elif frId > 300 and frId < 1000:
            ND = join(fn_ntwx, "leuven_models", "scriptData")
            mvIdx = "mv" + str(frId - 300)

            self.fdrs[frId] = mvIdx + "_lvn"  # leuven
            self.fdrsLoc[frId] = join(mvIdx, "master_1ph_" + mvIdx + "_slesNtwk.dss")

            # for the leuven models, copy the master script, then
            # replace the 'cd' dir so that this loads properly.
            fnRaw = join(ND, mvIdx, "master_raw_1ph_" + mvIdx + ".dss")
            with open(fnRaw) as file:
                rawTxt = file.read()
            fnUse = join(ND, self.fdrsLoc[frId])
            with open(fnUse, "w") as file:
                file.write(rawTxt.replace("_dir_name_here_", ND))

        return self.fdrs[frId], join(ND, self.fdrsLoc[frId])

    @staticmethod
    def getLvnsFn(
        frId,
    ):
        """A function for finding filename and ckt info for LVNS circuit types."""
        if frId < 100 or frId > 300:
            raise Exception("frId needs to be between 100 and 300.")

        ND0 = join(fn_ntwx, "manchester_models")

        dnA = "batch_manc_ntwx"
        dnB = "batch_manc_ntwx_mod"

        lvId = ((frId - 1) % 25) + 1
        i0 = frId - lvId
        iLv = str(lvId)

        # NB - as of 21/7/21, these could be tweaked still to get all working
        lvInfo = {  # frid0: [root, feederId end, master name end]
            100: [
                dnA,
                "",
                "",
            ],
            200: [
                dnA,
                "_p",
                "_pruned",
            ],
            225: [
                dnA,
                "",
                "_pruned_ukgds",
            ],
            250: [
                dnB,
                "_p_mod",
                "_mod",
            ],
            275: [dnA, "", "_pruned_one"],
        }

        ND = join(ND0, lvInfo[i0][0])
        feederId = "n" + iLv + lvInfo[i0][1]
        masterFn = join("network_" + iLv, "masterNetwork" + iLv + lvInfo[i0][2])

        return ND, feederId, masterFn

    def calcIprms(self, V=None):
        """Calculate the primitive currents for PDEs.

        Essentially does:
        IprmSet = []
        for yps in self.Yprm:
            IprmSet.append( yps['YprmI'].dot(V[yps['yzIdx']]) )
        IprmSet = np.concatenate(IprmSet)

        The equivalent dssIfc command is getIprm.
        """
        if V is None:
            V = np.r_[self.sln.VsrcMeas, self.sln.Vc]

        IprmSet = self.YprmBd.dot(V)
        return IprmSet

    def getYprmBd(
        self,
        **kwargs,
    ):
        """Get the block-diagonal BD primitive Y to go from V to Iprm.

        -----
        outputs
        -----
        Sets the following attributes:
        - YprmBd
        - YprmNms
        - YprmFlws
        - YprmBdKwargs
        - YprmBdCapTap

        If dssType is passed in, * is replaced with *_d.getObjName(dssType)

        -----
        kwargs
        -----
        prmType - flag is either 'YprmI' (all currents) or 'YprmV'
        dssType - is used to specify a type of element (e.g. d.TRN)
        psConv - flag indicates whether to return only pos seq admittances.
        capTap - the capTap position to get the YprmBd for

        When using psConv:
        - For three-phase transformers, the positive sequence current is found
          for winding 1.
        - For single-phase transformers, the current of winding 1 is used.
        """
        prmType = kwargs.get("prmType", "YprmI")
        dssType = kwargs.get("dssType", None)
        psConv = kwargs.get("psConv", False)
        capTap = kwargs.get("capTap", self.YbusCapTap)

        # First check if the captap positions have changed.
        capTapNm = "YprmBdCapTap"
        if dssType is not None:
            nm = d.getObjName(dssType)
            capTapNm = capTapNm + "_" + nm

        if getattr(self, capTapNm, None) == capTap:
            return

        # Then make sure we have the right Yprm:
        YbusKwargs_ = self.YbusKwargs
        self.getYset(yset="Yprm", fixCtrl=True, capTap=capTap)

        # Get the keys to loop through
        if dssType is None:
            keys = self.Yprm.keys()
        else:
            keys = []
            i = dssType.First
            while i:
                keys.append(d.AE.Name)
                i = dssType.Next
            # also, for ease we ignore transformers with regulators.
            if nm == "TRN":
                regXfm = d.getRegXfms()
                keys = [key for key in keys if key not in regXfm]

        # Loop through and pull out the required info:
        YprmBd_ = []
        yzIdxs = []
        emntNms = []
        flwNms = []
        for eName in keys:
            yp = self.Yprm[eName]
            yzIdxs.append(yp["yzIdx"])
            emntNms.append(eName)
            ypPrm = yp[prmType].copy()

            flwNms.extend(self.getFlwNms(yp, prmType, psConv))
            if not psConv:
                YprmBd_.append(ypPrm)
            else:
                if yp["nPh"] == 1:
                    YprmBd_.append(ypPrm[0].reshape(1, -1))
                elif yp["nPh"] == 3:
                    YprmBd_.append(phs2seq[1].dot(ypPrm[:3]).reshape(1, -1))
                else:
                    raise Exception("Only 1 & 3 phase xfmrs implemented.")

        # Create the YprmBd matrix and shift the voltage indices
        YprmBd = sparseBlkDiag(YprmBd_).tocsr()
        yzIdxs = np.concatenate(yzIdxs)

        idxShuff = sparse.coo_matrix(
            (np.ones(len(yzIdxs)), (np.arange(len(yzIdxs)), yzIdxs)),
            shape=(len(yzIdxs), self.nV + 3),
            dtype=int,
        ).tocsr()

        # List the new attributes to be set:
        newAttrs = {
            "YprmBd": YprmBd.dot(idxShuff),
            "YprmNms": emntNms,
            "YprmFlws": flwNms,
            "YprmBdKwargs": kwargs,
            "YprmBdCapTap": d.getCapTap(),
        }

        if dssType is not None:
            nm = d.getObjName(dssType)
            newAttrs = dict((key + "_" + nm, val) for key, val in newAttrs.items())

        # Update the setattrs as required:
        self.dictSetAttr(newAttrs)

        # Return the Ybus to the state it came in in:
        self.getYset(**YbusKwargs_)

    @staticmethod
    def getFlwNms(yp, prmType="YprmI", psConv=False):
        """Return list of flow names for the yp element.

        Inputs:
        - prmType is either 'YprmI' (default) or 'YprmV'
        - If psConv, then only the 'positive sequence name' is found.
        """
        if not psConv:
            if prmType == "YprmI":
                return [yp["eName"] + "__" + busSel for busSel in yp["iIds"]]
            elif prmType == "YprmV":
                return [yp["eName"] + "__" + busSel for busSel in yp["vIds"]]
        else:
            return [yp["eName"] + "__" + "posSeq"]

    def getYlli(self, capTap, **kwargs):
        """Find Ybus inverse, Ylli.

        If no existing inverse exists, first create the inverse and save.
        If it does exist, get it, and then update according to the taps/caps
        positions, as necessary.

        capTap: if set to None, forces a recalculationg of Ylli using the
                    current Ybus
        -----
        kwargs
        -----
        - force: relcalculates Ylli using state0's capTap
        - forceSave: recalculate Ylli using capTap
        - testYlli: compare to the full ybus.
        """
        force = kwargs.get("force", False)
        forceSave = kwargs.get("forceSave", False)
        testYlli = kwargs.get("testYlli", False)

        C = "Ylli"
        fn = self.getCacheFn(C)
        if capTap is None:
            Ylli_ = spla.inv(self.Ybus[3:, 3:]).toarray()
            capTap_ = self.YbusState["capTap"]
        elif force:
            self.getYset(yset="Ybus", fixCtrl=True, capTap=capTap)
            Ylli_ = spla.inv(self.Ybus[3:, 3:]).toarray()
            capTap_ = self.YbusState["capTap"]
        elif forceSave or (not os.path.exists(fn)):
            print("Building and saving Ylli...")
            Ylli_ = spla.inv(self.Ybus[3:, 3:]).toarray()
            capTap_ = self.YbusState["capTap"]
            dNet.saveCache(fn, {"Ylli": Ylli_, "capTap": capTap_}, cache=self.caches[C])
        elif "Ylli" in self.__dict__.keys():
            Ylli_ = self.Ylli
            capTap_ = self.YlliCapTap
        else:
            YlliRes = dNet.loadCache(fn, cache=self.caches[C])
            Ylli_ = YlliRes["Ylli"]
            capTap_ = YlliRes["capTap"]

        if capTap is None or capTap_ == capTap or force:
            self.Ylli = Ylli_
        else:
            self.Ylli = self.updateYlli(Ylli_, capTap_, capTap)
            capTap_ = capTap

        self.YlliCapTap = capTap_

        if testYlli:
            self.getYset(yset="Ybus", fixCtrl=True, capTap=capTap)
            Ylli_ = spla.inv(self.Ybus[3:, 3:]).toarray()
            val = np.linalg.norm(self.Ylli - Ylli_) / np.linalg.norm(Ylli_)
            print("Ylli Equal:", val < 1e-12)

    def updateYlli(self, Ylli_, capTap_, capTap):
        """Update admittance inverse to the positions of capTap."""

        (YmatOld, YmatNew) = self.getOldNewYmats(capTap_, capTap)[0]
        dYmat = (YmatNew - YmatOld)[3:, 3:]
        dYmat.eliminate_zeros()

        if dYmat.nnz == 0:
            raise Exception("dYmat=0 --> not calculated correctly!")

        return sparseInvUpdate(Ylli_, dYmat)

    @property
    def capTapTest(self):
        if not self.feeder in self.capTapTests.keys():
            raise Exception("Feeder " + self.feeder + " not in capTapTests.")

        return self.capTapTests[self.feeder]

    def test_getYlli(self):
        """A function for testing low-rank updating functions."""
        self.getYlli(capTap=self.state0["capTap"])
        Ylli0 = self.Ylli
        CapTap0 = self.YlliCapTap

        self.getYlli(capTap=self.capTapTest)
        Ylli = self.Ylli
        CapTap_ = self.YlliCapTap

        self.getYlli(capTap=self.capTapTest, force=True)
        YlliF = self.Ylli
        CapTapF = self.YlliCapTap
        rerr(Ylli, YlliF)

        print(CapTap0)
        print(self.capTapTest)
        print(CapTap_)
        print(CapTapF)

    def getCacheFn(self, C, getDn=False, prms={}):
        """Get the fn for cached values of C

        Inputs
        ---
        C: cache type
        getDn: if True, only returns the directory for C
        prms: parameters for modifying fn. Choose LoadMult for Vmlv.

        Options for C (corresponding to self.caches) are:
        - 'Ylli': inverse admittance matrices
        - 'Ybus': admittance matrices [not in use]
        - 'Vmlv': linear model for MLV linear models
        - 'cktInfo': bits and pieces to avoid having to recall OpenDSS

        Use 'getDn' flag to only get the directory of the files.
        """
        dn = os.path.join(fn_root, "lin_models", self.caches[C]["dn_"])
        if not getDn:
            if C in [
                "Ybus",
                "Ylli",
                "cktInfo",
            ]:
                fn = os.path.join(dn, self.caches[C]["fn_"] + self.feeder + ".pkl")
            elif C in ["Vmlv"]:
                lm = prms["LoadMult"]
                fn = os.path.join(
                    dn,
                    self.caches[C]["fn_"]
                    + self.feeder
                    + "_"
                    + f"{lm:.3f}".replace(".", "-")
                    + ".pkl",
                )
        else:
            fn = dn
        return fn

    @staticmethod
    def loadCache(fn, cache=None, vbs=True):
        """Load res from fn and return.

        If cache is set (e.g. as self.cache[C]) then check the keys align
        with the keys that should be loaded. Otherwise does no error checking.

        Use vbs flag to show/not print the data being loaded in.
        """
        with open(fn, "rb") as file:
            res = pickle.load(file)
            if vbs:
                print("Data loaded from\n\t-->" + fn)

        if not res is None:
            if set(res.keys()) != cache["res_"]:
                raise Exception("Data loaded from cache not valid.")
        return res

    @staticmethod
    def saveCache(fn, Res, cache=None, vbs=True):
        """Save Res to fn, with error checking using C (if provided).

        For options for cache, see e.g. help(self.getCacheFn) or self.caches.
        """
        if not cache is None:
            if cache["res_"] != set(Res.keys()):
                raise Exception("Data saving to cache not valid.")

        with open(fn, "wb") as file:
            pickle.dump(Res, file)
            if vbs:
                print("Data saved to\n\t-->" + fn)

    def getUbMats(self, test=False):
        """Get voltage unbalance matrices and indexes.

        Gets:
        - seqBus, the order of the three phase buses
        - seqBusPhs, the same but given unique names as to ps,ns,zs
        - Vc2Vcub, the mapping complex voltages to sequence voltages
        - nVseq, total number of sequence voltages
        - nVps, the total number of positive sequence voltages
        """
        (self.seqBus, self.seqBusPhs, self.vKvbaseSeq), self.Vc2Vcub = d.getVc2Vcub(
            self.YZidxs
        )
        self.nVseq = len(self.seqBusPhs)
        self.nVps = len(self.seqBus)
        if test:
            self.testVc2Vcub()

    def testVc2Vcub(self):
        """Test the voltage unbalance model against OpenDSS metrics."""
        self.getSln(self.state0)
        Vcub = self.Vc2Vcub.dot(np.r_[self.sln.VsrcMeas, self.sln.Vc])
        VcubDss = d.getVseq(seqBus=self.seqBus)
        print("Vcub error: {:.6g}".format(norm(VcubDss - Vcub) / norm(VcubDss)))

    # Index shuffling ======
    def xydt2x(self, YY, DD, TT, dim=1):
        """Convery YY, DD, YY triple to a single object XX.

        Recall that X is of the form as given in state2x.

        The opposite of this is x2xydt.
        """
        if YY.ndim == 1:
            if YY.shape[0] == self.nV * 2:
                YY = YY[self.syIdx]
                DD = DD[self.sdIdx]

            if YY.shape[0] == self.nPy * 2:
                XX = np.r_[
                    YY[: self.nPy], DD[: self.nPd], YY[self.nPy :], DD[self.nPd :], TT
                ]
        if YY.ndim == 2:
            if dim == 0:
                XX = np.r_[
                    YY[: self.nPy], DD[: self.nPd], YY[self.nPy :], DD[self.nPd :], TT
                ]
            elif dim == 1:
                XX = np.c_[
                    YY[:, : self.nPy],
                    DD[:, : self.nPd],
                    YY[:, self.nPy :],
                    DD[:, self.nPd :],
                    TT,
                ]
        return XX

    def x2xydt(
        self,
        XX,
        xs=None,
        nPydt=None,
    ):
        """Convert XX to individual control elements in wye, delta, tap ctrls.

        The opposite of this is xydt2x.
        - Pass in xs (int) if splitting a matrix.
        - Pass in nPdyt [nPy,nPd,nT] for other dimensions if required
        """
        # Get the indices, if not the nominal ones for this circuit
        if nPydt is None:
            nPy = self.nPy
            nPd = self.nPd
            nT = self.nT
        else:
            nPy, nPd, nT = nPydt

        nP = nPy + nPd
        nS = nP * 2
        nCtrl = nS + nT

        # If only a vector is passed, split on the first axis:
        if xs is None and XX.ndim == 1:
            xs = 0

        # Check if an xs has been passed, if required:
        if xs is None:
            raise Exception("Please pass in an axis to iterate over!")

        # Check the dimensions of XX are as expected:
        if XX.shape[xs] != (nCtrl):
            raise Exception("XX is not of conformal dimensions!")

        # Finally pull out the matrices as requested:
        if xs == 0:
            YY = np.r_[XX[:nPy], XX[nP : nP + nPy]]
            DD = np.r_[XX[nPy:nP], XX[nP + nPy : nS]]
            TT = XX[nS:]
        if xs == 1:
            YY = np.c_[XX[:, :nPy], XX[:, nP : nP + nPy]]
            DD = np.c_[XX[:, nPy:nP], XX[:, nP + nPy : nS]]
            TT = XX[:, nS:]

        return YY, DD, TT

    def x2xc(self, X, conj=False):
        """Convert real X to complex X, plus tap controls."""
        if conj:
            val = np.r_[
                X[: self.nP] - 1j * X[self.nP : self.nS], X[self.nS : self.nS + self.nT]
            ]
        else:
            val = np.r_[
                X[: self.nP] + 1j * X[self.nP : self.nS], X[self.nS : self.nS + self.nT]
            ]
        return val

    def state2xc(
        self,
        state,
        conj=False,
    ):
        """A simple wrapper for self.x2xc(self.state2x(state))."""
        return self.x2xc(self.state2x(state), conj)

    def getXc(self):
        """Alias of self.x2xc(self.getX())

        Also equivalent to self.x2xc( self.state2x( d.getNtwkState() ) )
        """
        return self.x2xc(self.getX())

    @staticmethod
    def getVolLevels(obj):
        obj.lvIdx = np.where(obj.vKvbase < 1e3)[0]
        obj.nVlv = len(obj.lvIdx)
        obj.mvIdx = np.where(obj.vKvbase > 1e3)[0]
        obj.nVmv = len(obj.mvIdx)

    def state2x(
        self,
        state,
    ):
        """Process a dssIfc state into a control X.

        Recall: X is of the form
                            X = [Py Pd Qy Qd T]

        The injection indexes are given by [possibly???]
        - Py / Qy: self.pyIdx
        - Pd / Qd: self.pdIdx (corresponding to the first phase)
        - T:  the Xfms in d.getRegXfms()

        To go from state (indexes in d.LDS, d.GEN), a list of powers is
        created from state LDS, GEN in that order, for both Y and D loads.

        The indexes for things in X are given by syIdx,sdIdx for powers.

        """
        # Create a vector of power injections from loads and gens:
        kLds = -state["LoadMult"] * 1e3
        kGen = 1e3
        injY = np.r_[kLds * np.array(state["LdsY"]), kGen * np.array(state["GenY"])]
        injD = np.r_[kLds * np.array(state["LdsD"]), kGen * np.array(state["GenD"])]

        # Put these into a longer list that specifies injections per-phase:
        nZy = self.nPyLds + self.nPyGen
        nZd = self.nPdLds + self.nPdGen

        xYlist = np.zeros(nZy, dtype=complex)
        xDlist = np.zeros(nZd, dtype=complex)
        i = 0
        for nPhs, inj in zip(self.iIdxPhsY, injY):
            xYlist[i : i + nPhs] = inj / nPhs
            i += nPhs

        i = 0
        for nPhs, inj in zip(self.iIdxPhsD, injD):
            xDlist[i : i + nPhs] = inj / nPhs
            i += nPhs

        idxY = np.r_[self.LDS_.Y.idx, self.GEN_.Y.idx]
        idxD = np.r_[self.LDS_.D.idx, self.GEN_.D.idx]

        # A little hack to add any powers together at the same node:
        xY0 = sparse.coo_matrix((xYlist, (idxY, np.zeros(nZy))), [self.nV, 1])
        xD0 = sparse.coo_matrix((xDlist, (idxD, np.zeros(nZd))), [self.nV, 1])

        Xy = xY0.toarray().ravel()
        Xy = np.r_[Xy.real, Xy.imag]
        Xd = xD0.toarray().ravel()
        Xd = np.r_[Xd.real, Xd.imag]

        # Taps are a bit messier:
        if not hasattr(self, "mdl"):
            # Xt = np.nan*np.zeros(self.nT) # WARNING < not sure why nans here?
            Xt = np.zeros(self.nT)
        else:
            Xt = np.array(state["capTap"][1]) - self.mdl.linTap

        return self.xydt2x(Xy, Xd, Xt)

    def getX(self):
        """Alias of self.state2x( d.getNtwkState() )"""
        return self.state2x(d.getNtwkState())

    def getSln(self, state, uRcrd=False, uRcrdIdx=None):
        """Find a solution at state, then save as sln struct.

        Based on "TL,PL,TC,CL,V,I,Vc,Ic = self.slnF" from lineariseDssModels

        Pass in state=None to initialise this.

        ====
        Inputs
        ====
        state: required, the state at which the solution is found
        uRcrd: whether or not to also update the update record slnr
        uRcrdIdx: If saving with uRcrd, saves only these idxs

        ====
        Creates
        ====
        sln: the solution struct, containing
            - stateIn, the state used to solve opendss
            - VsrcMeas, the voltages at the source solution
            - Vc, the voltages excluding the source
            - stateOut, the state after solving (taps etc may change)
            - vc, V, v; derived voltages in magnitude/pu etc
            - TP, total power
            - TL, total losses
            - Cnvg, converged flag
        """
        if state is None:
            # Initiliase if required
            self.sln = struct()
            for key, val in self.slnVals.items():
                setattr(self.sln, key, val)
        else:
            # Otherwise return by setting the values from slnVals.
            self.sln.stateIn = deepcopy(state)
            VcAll = d.setNtwkState(state)
            self.sln.VsrcMeas = VcAll[:3]
            self.sln.Vc = VcAll[3:]
            self.sln.stateOut = d.getNtwkState()

            getVset(self.sln, self.vKvbase)  # gets V,vc,v

            self.sln.TP = tp2ar(d.DSSCircuit.TotalPower)[0]  # kW
            self.sln.TL = tp2ar(d.DSSCircuit.Losses)[0] / 1e3  # kW
            self.sln.Cnvg = d.SLN.Converged

        if uRcrd:
            self.updateRecord(uRcrdIdx)

    def updateRecord(self, uRcrdIdx=None):
        if not hasattr(self, "slnr"):
            self.slnr = struct()
            self.slnr.stateIn = []
            self.slnr.stateOut = []
            self.slnr.TP = np.empty((0))  # so .real works
            self.slnr.TL = np.empty((0))
            self.slnr.Cnvg = []
            self.slnr.VsrcMeas = np.empty((0, 3))

            if uRcrdIdx is None:
                self.slnr.uRcrdIdx = np.arange(self.nV)
                self.slnr.YZv = self.YZv
                self.slnr.vKvbase = self.vKvbase
                self.slnr.Vc = np.empty((0, self.nV))  # run getVset() to get all
            else:
                self.slnr.uRcrdIdx = uRcrdIdx
                self.slnr.YZv0 = self.YZv
                self.slnr.YZv = vecSlc(self.YZv, uRcrdIdx)
                self.slnr.vKvbase = self.vKvbase[uRcrdIdx]
                self.slnr.Vc = np.empty((0, len(uRcrdIdx)))

        self.slnr.stateIn.append(self.sln.stateIn)
        self.slnr.stateOut.append(self.sln.stateOut)

        self.slnr.TP = np.r_[self.slnr.TP, self.sln.TP]
        self.slnr.TL = np.r_[self.slnr.TP, self.sln.TL]
        self.slnr.Cnvg.append(self.sln.Cnvg)
        self.slnr.VsrcMeas = np.r_[self.slnr.VsrcMeas, np.array([self.sln.VsrcMeas])]

        if uRcrdIdx is None:
            self.slnr.Vc = np.r_[self.slnr.Vc, np.array([self.sln.Vc])]
        else:
            self.slnr.Vc = np.r_[self.slnr.Vc, np.array([self.sln.Vc[uRcrdIdx]])]

    @staticmethod
    def cMat2rMat(cMat, pmConj=False):
        """Convert complex matrix to 2x2 real counterpart.

        If pmConj is False, is a 'normal' complex to real matrix,
        [ [R,-I],[I,R] ]. If true, is [ [R,I],[I,-R] ] (and
        cannot be represented by a single complex number).

        Can be np arrays or sparse csc/csr matrices.
        """
        if type(cMat) is np.ndarray:
            if not pmConj:
                rMat = np.block([[cMat.real, -cMat.imag], [cMat.imag, cMat.real]])
            else:
                rMat = np.block([[cMat.real, cMat.imag], [cMat.imag, -cMat.real]])
        if type(cMat) in [sparse.csr.csr_matrix, sparse.csc.csc_matrix]:
            if not pmConj:
                rMat = sparse.bmat([[cMat.real, -cMat.imag], [cMat.imag, cMat.real]])
            else:
                rMat = sparse.bmat([[cMat.real, cMat.imag], [cMat.imag, -cMat.real]])
        return rMat

    # =============================== Plotting Functions
    def getBusPhs(self):
        self.bus0v, self.phs0v = self.busPhsLoop(self.YZv)
        self.bus0s, self.phs0s = self.busPhsLoop(self.YZsy + self.YZsd)

        # # Legacy
        # self.bus0vIn, self.phs0vIn = self.busPhsLoop(self.vInYNodeOrder)
        # self.bus0vIn, self.phs0vIn = self.busPhsLoop(self.vInYNodeOrder)

    def busPhsLoop(self, YZset):
        bus0 = []
        phs0 = []
        for yz in YZset:
            fullBus = yz.split(".")
            bus0.append(fullBus[0].lower())
            if len(fullBus) > 1:
                phs0.append(fullBus[1::])
            else:
                phs0.append(["1", "2", "3"])
        return np.array(bus0), np.array(phs0)

    def getSourceBus(self):
        d.Vsrcs.First
        self.vSrcBus = d.AE.BusNames[0]

    def setupPlots(self):
        """Get data required for using plotNetwork.

        Creates
        ---
        self.busCoords - dict of bus:(x,y)
        self.branches - odict of brch:[bus0,bus1]
        self.regBuses - list of buses which are regulated
        self.srcReg - flag if there is a regulator at the source node
        - also updates fPrm_['pms'] to a new size if wanted.
        - Then runs self.getBusPhs() to set
            > self.bus0v, self.phs0v, self.bus0s, self.phs0s, the buses/phases
              for different plotting types.

        """
        self.busCoords = d.getBusCoordsAug()
        self.branches = odict(d.getBranchBuses())
        self.regBuses = d.getRgcNds(self.YZvIdxs)[0]

        # if there is a regulator 'on' the source bus
        regSrc = [
            "epri24",
            "8500node",
            "123bus",
            "123busCvr",
            "epriJ1",
            "epriK1",
            "epriM1",
        ]
        self.srcReg = 1 if self.feeder in regSrc else 0

        # Update fPrm_['pms'] if wanted.
        plotMarkerDict = {
            "13bus": 150,
            "34bus": 150,
            "eulv": 100,
            "epri5": 10,
            "epriK1cvr": 15,
            "epriK1": 15,
            "8500node": 8,
        }

        if self.feeder in plotMarkerDict.keys():
            self.fPrm_["pms"] = plotMarkerDict[self.feeder]
        elif self.feeder[0] == "n":
            self.fPrm_["pms"] = 40

        # Get the bus and phase information for voltage/powers.
        self.getBusPhs()

    def plotNetwork(self, pType=None, **kwargs):
        """Plot a network with option pType and colorbar minMax vals.

        Adapted from plotNetBuses from lineariseDssModels

        -----
        INPUTS
        -----
        pType - an option to choose what thing to plot.

            --- 'Node' options:
            > None  - a basic plot with all colours identical
            > 'n'   - a basic plot with all colours identical
            > 'v'   - mean node voltage
            > 'vPh' - phase voltages
            > 'p'   - total real power injection
            > 'q'   - total reactive power injection
            > 's'   - total apparent power injection
            > 'qPh' - daisy plot of reactive powers

            --- 'Branch' options:
            > '_s' - power through each line
            > '_sLog' - same, but log taken
            > '_L' - losses in the line

        kwargs:
            minMax  - the min/max value for the colorbar, if there is one.
            score0  - force a score value to be passed into get bus_score
            cmapSet - force the colormap color to cmapSet
            cmapSetB - force the branch colormap color to cmapSetB
            cbTtl   - force the colormap title
            ax      - axis to plot onto

        """
        cmapSet = kwargs.get("cmapSet", None)
        bMinMax = kwargs.get("bMinMax", None)
        ax = kwargs.get("ax", None)
        pType = "n" if pType is None else pType

        if not hasattr(self, "busCoords"):
            self.setupPlots()

        if ax is None:
            _, ax = plt.subplots()

        # First plot the branches:
        if pType[0] != "_":
            self.plotBranches(ax)
        else:
            bch_scores, bMinMax = self.get_bch_score(
                pType,
                bMinMax,
            )
            self.plotBranches(
                ax,
                scores=bch_scores,
                kw=kwargs,
            )

            # Opts: cmsq(cm.PuBuGn,0.8,0.9),
            cmapSet = cmsq(cm.Greys, 0.6, 0.7) if cmapSet is None else cmapSet

        # self.plotSub(ax,pltSrcReg=False)
        self.plotSub(ax)

        ttl, edgeOn, pltRegs, busNans, cbTtl0, bus_score, minMax0 = self.get_bus_score(
            pType,
            kwargs,
        )

        # Set the cmaps for plotting
        if not cmapSet is None:
            self.fPrm_["cmap"] = cmapSet
        elif pType[0] in self.fPrm_["cms"].keys():
            self.fPrm_["cmap"] = self.fPrm_["cms"][pType[0]]
        else:
            self.fPrm_["cmap"] = plt.cm.viridis

        # Get the minmax values
        minMax = kwargs.get("minMax", minMax0)
        cbTtl = kwargs.get("cbTtl", cbTtl0)

        # Plot the buses
        if pType[-2:] == "Ph":
            self.plotBuses3ph(ax, bus_score, minMax)
        else:
            self.plotBuses(ax, bus_score, minMax, edgeOn=edgeOn, busNans=busNans)

        # Plot the colorbar
        if (
            pType
            not in [
                "n",
                "B",
            ]
            and pType[0] != "_"
        ):
            self.plotNetColorbar(ax, minMax, cbTtl=cbTtl)

        # Plot the regs
        if pltRegs:
            self.plotRegs(ax)

        # Tidy up the figure & set title
        ax.axis("off")
        if self.fPrm_["ttlOn"]:
            plt.title(self.feeder, loc="left")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(False)
        plt.tight_layout()

        # Save figure if we like using sff_legacy, fPrm
        figname = whocalledme(2) + "_" + self.feeder + "_" + pType
        self.sff_(figname)

    def get_bch_score(
        self,
        pType,
        bMinMax=None,
    ):
        """Get the branch score to pass into plotBranches.

        Inputs
        ---
        pType - the plot type
        bMinMax - the max & min plotting scale

        Returns
        ---
        scores, values between 0 (or 0.1) and 1 for each bus
        bMinMax, the corresponding max and min values
        """
        nm2ii = {nm: i for i, nm in enumerate(d.getObjAttr(d.PDE, AEval="Name"))}

        if pType in ["_s", "_sLog"]:
            pwrs = tp2arL(d.getObjAttr(d.PDE, AEval="Powers"))
            scores = np.array(
                [
                    sum(np.abs(pwrs[nm2ii[br]])[: self.Yprm[br]["nAI"]])
                    for br in self.branches.keys()
                ]
            )

            if pType == "_sLog":
                scores = np.log10(scores)
                bMinMax = (
                    [min(scores[scores > (max(scores) - 6)]), max(scores)]
                    if bMinMax is None
                    else bMinMax
                )
            else:
                bMinMax = (
                    [1e-6 * max(scores), max(scores)] if bMinMax is None else bMinMax
                )

        elif pType == "_L":
            lsss = [r[0] for r in d.getObjAttr(d.PDE, AEval="Losses")]
            scores = np.array([lsss[nm2ii[br]] for br in self.branches.keys()])

            bMinMax = [1e-10 * max(scores), max(scores)] if bMinMax is None else bMinMax

        # Remove small values
        scores[scores < bMinMax[0]] = np.nan

        # Fit to min/max
        scores = (scores - bMinMax[0]) / (bMinMax[1] - bMinMax[0])

        return scores, bMinMax

    def get_bus_score(
        self,
        pType,
        kwargs,
    ):
        """Get the bus score (and other properties) based on pType.

        Inputs
        ---
        pType - the plot type, see help(self.plotNetwork)
        kwargs - the kwargs passed into self.plotNetwork.

        Returns
        ---
        ttl - the title for the figure
        edgeOn - flag for buses to have edges or not on the lines
        pltRegs - flag to plot voltage regulators
        busNans - flag to plot buses with value of nan or not
        cbTtl0 - title for the colorbar
        bus_score - score for the buses to be plotted
        minMax0 - minimum and maximum values to scale the scores by
        """

        ttl = None
        edgeOn = True
        pltRegs = True
        busNans = True

        if pType == "n":
            bus_score0 = kwargs.get("score0", np.ones((self.nP)))
            bus_score, minMax0 = self.getScrVals(bus_score0, "s", "mean")
            edgeOn = False
            cbTtl0 = ""
        elif pType == "v":
            bus_score0 = kwargs.get("score0", np.abs(self.sln.V) / self.vKvbase)
            bus_score, minMax0 = self.getScrVals(bus_score0, "v", "mean")
            cbTtl0 = "Voltage, pu"
        elif pType == "B":
            # Build plot - based on Turing code
            bus_score0 = kwargs["score0"]  # required to be passed in
            bus_score, _ = self.getScrVals(bus_score0, "v", "mean")
            minMax0 = kwargs["minMax0"]
            pltRegs = False
            cbTtl0 = ""
        elif pType == "vPh":
            bus_score0 = kwargs.get("score0", np.abs(self.sln.V) / self.vKvbase)
            bus_score, minMax0 = self.getScrVals(bus_score0, "v", "ph")
            cbTtl0 = "Voltage, pu"
        elif pType == "p":
            bus_score0 = kwargs.get("score0", -1e-3 * self.getX()[: self.nP])
            bus_score, minMax0 = self.getScrVals(bus_score0, "s", "sum")
            cbTtl0 = "Load, kW"
        elif pType == "q":
            bus_score0 = kwargs.get("score0", -1e-3 * self.getX()[self.nP : self.nS])
            bus_score, minMax0 = self.getScrVals(bus_score0, "s", "sum")
            minMaxAbs = np.max(
                np.abs(
                    np.array(
                        [
                            np.nanmax(list(bus_score.values())),
                            np.nanmin(list(bus_score.values())),
                        ]
                    )
                )
            )
            minMax0 = [-minMaxAbs, minMaxAbs]
            cbTtl0 = "$Q$, kVAr"  # Positive implies capacitive
        elif pType == "s":
            bus_score0 = kwargs.get("score0", 1e-3 * np.abs(self.getXc()[: self.nP]))
            bus_score, minMax0 = self.getScrVals(bus_score0, "s", "sum")
            cbTtl0 = "$|S|$, kVA"
        elif pType == "qPh":
            bus_score0 = kwargs.get("score0", -1e-3 * self.getX()[self.nP : self.nS])
            bus_score, minMax0 = self.getScrVals(bus_score0, "s", "ph")
            scoresFull = (
                list(bus_score["1"].values())
                + list(bus_score["2"].values())
                + list(bus_score["3"].values())
            )
            minMaxAbs = np.max(
                np.abs(np.array([np.nanmax(scoresFull), np.nanmin(scoresFull)]))
            )
            minMax0 = [-minMaxAbs, minMaxAbs]
            cbTtl0 = "$Q$, kVAr"  # Positive implies capacitive
            # self.fPrm_['tFs'] = 8 # ttlFontsize
        elif pType[0] == "_":
            bus_score0 = kwargs.get("score0", np.ones((self.nP)))
            bus_score, minMax0 = self.getScrVals(bus_score0, "s", "mean")
            edgeOn = False
            cbTtl0 = ""
            busNans = False

        return ttl, edgeOn, pltRegs, busNans, cbTtl0, bus_score, minMax0

    def plotNetColorbar(self, ax, minMax, cbTtl=None, nCbar=150):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = [xlim[0] - 1]
        y0 = [ylim[0] - 1]
        cntr = ax.contourf(
            np.array([x0 * 2, x0 * 2]),
            np.array([y0 * 2, y0 * 2]),
            np.array([minMax, minMax[::-1]]),
            nCbar,
            cmap=self.fPrm_["cmap"],
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # cbar = plt.colorbar(cntr,shrink=0.75,
        # ticks=np.linspace(minMax[0],minMax[1],5))
        cbar = plt.colorbar(
            cntr, shrink=0.6, ticks=np.linspace(minMax[0], minMax[1], 5)
        )
        cbar.ax.tick_params(labelsize=self.fPrm_["tFs"])
        if cbTtl != None:
            cbar.ax.set_title(cbTtl, pad=10, fontsize=self.fPrm_["tFs"])

    def plotBuses(
        self,
        ax,
        scores,
        minMax,
        edgeOn=True,
        busNans=True,
    ):
        """Plot the buses using scores scores.

        Inputs
        ---
        ax - the axis
        scores - the sscores to uses
        minMax - linearly scale between minMax (useful for extending range)
        edgeOn - if True, puts a thin edge around each bus with a score
        busNans - if True, plots the busNans with color '#cccccc'.

        """
        cmap = plt.cm.viridis if self.fPrm_["cmap"] is None else self.fPrm_["cmap"]

        x0scr, y0scr, xyClr, x0nne, y0nne = mtList(5)
        for bus, coord in self.busCoords.items():
            if not np.isnan(self.busCoords[bus][0]):
                if np.isnan(scores[bus]):
                    x0nne.append(coord[0])
                    y0nne.append(coord[1])
                else:
                    x0scr.append(coord[0])
                    y0scr.append(coord[1])
                    if minMax == None:
                        score = scores[bus]
                    else:
                        score = (scores[bus] - minMax[0]) / (minMax[1] - minMax[0])

                    xyClr.append(cmap(score))

        plt.scatter(
            x0scr, y0scr, marker=".", color=xyClr, zorder=+10, s=self.fPrm_["pms"]
        )
        if edgeOn:
            plt.scatter(
                x0scr,
                y0scr,
                marker=".",
                zorder=+11,
                s=self.fPrm_["pms"],
                facecolors="none",
                edgecolors="k",
            )

        if busNans:
            plt.scatter(x0nne, y0nne, color="#cccccc", marker=".", zorder=+5, s=3)

    def plotBuses3ph(self, ax, scores3ph, minMax, edgeOn=True):
        if self.fPrm_["cmap"] is None:
            cmap = plt.cm.viridis
        else:
            cmap = self.fPrm_["cmap"]
        x0scr = []
        y0scr = []
        xyClr = []
        x0nne = []
        y0nne = []
        phscr = []
        scale = 50
        phsOffset = {"1": [0.0, 1.0], "2": [-0.86, -0.5], "3": [0.86, -0.5]}
        for ph, scores in scores3ph.items():
            for bus, coord in self.busCoords.items():
                if not np.isnan(self.busCoords[bus][0]):
                    if np.isnan(scores[bus]):
                        x0nne = x0nne + [coord[0]]
                        y0nne = y0nne + [coord[1]]
                    else:
                        phscr = phscr + [ph]
                        x0scr = x0scr + [coord[0]]
                        y0scr = y0scr + [coord[1]]
                        if minMax == None:
                            score = scores[bus]
                        else:
                            score = (scores[bus] - minMax[0]) / (minMax[1] - minMax[0])

                        xyClr = xyClr + [cmap(score)]

            self.hex3phPlot(
                ax, x0scr, y0scr, xyClr, phscr, sf=self.sfDict3ph[self.feeder]
            )
            # plt.scatter(x0scr,y0scr,marker='.',color=xyClr,
            # zorder=+10,s=self.plotMarkerSize)
            # if edgeOn: plt.scatter(x0scr,y0scr,marker='.',
            # zorder=+11,s=self.plotMarkerSize,
            # facecolors='none',edgecolors='k')
            plt.scatter(x0nne, y0nne, color="#cccccc", marker=".", zorder=+5, s=15)

    def hex3phPlot(self, ax, x, y, xyClr, xyPhs, sf):
        xy0 = np.exp(1j * np.linspace(np.pi / 2, 5 * np.pi / 2, 7))
        # phsOffset =  {'1':np.exp(1j*np.pi/6),'2':np.exp(1j*5*np.pi/6),
        # '3':np.exp(1j*-np.pi/2)}
        phsOffset = {
            "1": np.exp(1j * 0),
            "2": np.exp(1j * 2 * np.pi / 3),
            "3": np.exp(1j * 4 * np.pi / 3),
        }
        patches = []
        for i in range(len(x)):
            brSf = 1.15
            xyPts = np.c_[
                sf * (xy0 + brSf * phsOffset[xyPhs[i]]).real + x[i],
                sf * (xy0 + brSf * phsOffset[xyPhs[i]]).imag + y[i],
            ]
            # NB putting zorder here doesn't seem to work vvvv
            patches.append(Polygon(xyPts, True, fill=1, color=xyClr[i]))
            patches.append(
                Polygon(
                    xyPts, True, fill=0, linestyle="-", linewidth=0.4, edgecolor="k"
                )
            )
            patches.append(
                Polygon(
                    [
                        [x[i], y[i]],
                        [
                            x[i] + sf * brSf * phsOffset[xyPhs[i]].real,
                            y[i] + sf * brSf * phsOffset[xyPhs[i]].imag,
                        ],
                    ],
                    False,
                    color="k",
                    linewidth=1.0,
                )
            )
        p = PatchCollection(patches, match_original=True)
        p.set_zorder(10)
        ax.add_collection(p)

    def plotRegs(self, ax):
        if self.nT > 0:
            i = 0
            for regBus in self.regBuses:
                regCoord = self.busCoords[regBus.split(".")[0].lower()]
                if not np.isnan(regCoord[0]):
                    ax.plot(regCoord[0], regCoord[1], "r", marker=(6, 1, 0), zorder=+5)
                    # ax.annotate(str(i),(regCoord[0],regCoord[1]),zorder=+40)
                else:
                    print("Could not plot regulator bus" + regBus + ", no coordinate")
                i += 1
        else:
            print("No regulators to plot.")

    def plotBranches(self, ax, scores=[], kw={}):
        """Plot the branches.

        To plot LV networks 'per feeder', then pass in lvFdrs=True

        Inputs
        ---
        ax - the axis to plot onto
        scores - the scores to plot with (if [] then plot as #cccccc
        kw - kwargs passed into plotNetwork

        """
        cmsel = kw.get("cMapSetB", cm.cividis_r)

        segclr, segnan, clrs = mtList(3)
        for i, buses in enumerate(self.branches.values()):
            # buses[:2] required due to, e.g., 3-winding transformers
            bcs = [self.busCoords[b.split(".")[0]] for b in buses[:2]]

            if len(scores) == 0:
                segnan.append(bcs)
            else:
                if np.isnan(scores[i]):
                    segnan.append(bcs)
                else:
                    segclr.append(bcs)
                    if not kw.get("lvFdrs", False):
                        clrs.append(cmsel(scores[i]))
                    else:
                        # Get the feeder no
                        try:
                            fdri = int(buses[1].split("_")[0][-1])
                        except:
                            fdri = 0

                        clrs.append(cmsel[fdri](scores[i]))

        if len(scores) == 0:
            coll = LineCollection(segnan, colors="#cccccc")
        else:
            # At the moment, ignore segnans for colored plots. Reverse also
            # seems a little cleaner for most of these.
            coll = LineCollection(segclr[::-1], colors=clrs[::-1])

        ax.add_collection(coll)
        ax.autoscale_view()

    def plotSub(self, ax, pltSrcReg=True):
        """Plot the substation onto axis ax."""
        srcCoord = self.busCoords[self.vSrcBus]
        if np.isnan(srcCoord[0]):
            print(
                "Nominal source bus coordinate not working," + " trying z appended..."
            )
            try:
                srcCoord = self.busCoords[self.vSrcBus + "z"]
            except:
                srcCoord = [np.nan, np.nan]
        if np.isnan(srcCoord[0]):
            print("Nominal source bus coordinate not working, trying 1...")
            srcCoord = self.busCoords["1"]

        if not np.isnan(srcCoord[0]):
            ax.plot(srcCoord[0], srcCoord[1], "k", marker="H", markersize=8, zorder=+20)
            ax.plot(srcCoord[0], srcCoord[1], "w", marker="H", markersize=3, zorder=+21)
            if self.srcReg and pltSrcReg:
                ax.plot(
                    srcCoord[0], srcCoord[1], "r", marker="H", markersize=3, zorder=+21
                )
            else:
                ax.plot(
                    srcCoord[0], srcCoord[1], "w", marker="H", markersize=3, zorder=+21
                )
        else:
            print("Could not plot source bus" + self.vSrcBus + ", no coordinate")

    def getScrVals(self, Scrs, busType="v", aveType=None):
        """Get the score values for each location in the busType."""
        if aveType is None:
            aveType = "mean"
        # direct copy
        if busType == "v":
            bus0 = self.bus0v
            phs0 = self.phs0v.flatten()
        elif busType in ["s", "p", "q", "n"]:
            bus0 = self.bus0s
            phs0 = self.phs0s.flatten()
        elif busType == "vIn":
            bus0 = self.bus0vIn
            phs0 = self.phs0vIn.flatten()

        scrMin = 1e100
        scrMax = -1e100
        scrVals = {}
        if aveType == "ph":
            scrVals["1"] = {}
            scrVals["2"] = {}
            scrVals["3"] = {}
            for bus in self.busCoords:  # initialise
                scrVals["1"][bus] = np.nan
                scrVals["2"][bus] = np.nan
                scrVals["3"][bus] = np.nan
        else:
            for bus in self.busCoords:
                scrVals[bus] = np.nan  # initialise

        for bus in self.busCoords:
            if not np.isnan(self.busCoords[bus][0]):
                if aveType == "ph":
                    phs = phs0[bus0 == bus.lower()]

                vals = Scrs[bus0 == bus.lower()]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    if aveType == "mean":
                        scrVals[bus] = np.mean(vals)
                        scrMax = max(scrMax, np.mean(vals))
                        scrMin = min(scrMin, np.mean(vals))
                    elif aveType == "sum":
                        scrVals[bus] = np.sum(vals)
                        scrMax = max(scrMax, np.sum(vals))
                        scrMin = min(scrMin, np.sum(vals))
                    elif aveType == "max":
                        scrVals[bus] = np.max(vals)
                        scrMax = max(scrMax, np.max(vals))
                        scrMin = min(scrMin, np.max(vals))
                    elif aveType == "min":
                        scrVals[bus] = np.min(vals)
                        scrMax = max(scrMax, np.min(vals))
                        scrMin = min(scrMin, np.min(vals))
                    elif aveType == "ph":
                        i = 0
                        for ph in phs:
                            scrVals[ph[0]][bus] = vals[i]
                            scrMax = max(scrMax, np.min(vals[i]))
                            scrMin = min(scrMin, np.min(vals[i]))
                            i += 1

        if scrMin == scrMax:
            scrMinMax = None
        else:
            scrMinMax = [scrMin, scrMax]
        return scrVals, scrMinMax

    def sff_(self, figName=None):
        """Alias for sff_legacy(self.fPrm,figName)."""
        sff_legacy(self.fPrm, figName)


class lNet(dNet):
    """A linearised network of the dNet - deprecated.
    """
    linDir = join(sys.path[0], "lin_models")

    lmPairs = {}
    pmlPairs = {}
    lmSets = {}

    def __init__(
        self,
        frId=5,
        tests=[],
    ):
        """Basic initialisation using dNet class, not using any tests."""
        dNet.__init__(
            self,
            frId,
            tests=tests,
        )

class mlvNet(lNet):
    """A derivative of the basic lNet, designed for large MV-LV systems.

    Based on the decomposition of
    V = Mx + a
    into the form
    V = Mx + CFx + a
    where M is sparse block-diagonal, C and F are 'thin' matrices, to reduce
    the computational overhead.

    Notes.
    - lmPairs and lmSets should override those variables from lNet

    """

    lmPairs = {
        "M1": ["i", "Vmlv", "C", "F"],
        "K1": ["j", "|Vmlv|", "D", "F"],
        # '_K1':['_j','_|Vmlv|','_D','_F'],
        "O": ["h", "V_mv"],
    }

    lmSets = {
        "all": list(lmPairs.keys()),
        "bareMv": [
            "O",
        ],
    }

    caches = dNet.caches
    caches.update(
        {
            "Vmlv": {
                "name": "Vmlv",
                "dn_": "VmlvCache",
                "fn_": "Ymlv_",
                "res_": {
                    "mdl",
                    "ckts",
                    "mlvIdx",
                    "mlvNsplt",
                    "mlvKvbase",
                },
            },
            "cktInfo": {
                "name": "cktInfo",
                "dn_": "cktInfo",
                "fn_": "cktInfo_",
                "res_": {
                    "cktInfo",
                },
            },
        }
    )

    # Set the names of the circuits for figures etc
    tidyCktNames = {
        90: "UG",
        91: "UG/OH, A",
        92: "UG/OH, B",
        93: "OH, A",
        94: "OH, B",
        95: "OH/UG, A",
        96: "OH/UG, B",
    }
    tidyCktNames.update({key - 10: val for key, val in tidyCktNames.items()})

    # cktInfo is a dict of circuit info to save to avoid having to
    # load/reload opendss where possible. Functions which this should allow:
    # - self.state2x
    # - self.xydt2x
    # - self.state2xc
    cktInfoHead = [
        "nPyLds",  # state2x
        "nPyGen",
        "nPdLds",
        "nPdGen",
        "iIdxPhsY",
        "iIdxPhsD",
        "LDS_",
        "GEN_",
        "nV",
        "nT",
        "syIdx",  # xydt2x
        "sdIdx",
        "nPd",
        "nPy",
        "nP",  # state2xc
        "nS",
    ]

    def __init__(
        self,
        frId=39,
        reloadDss=True,
    ):
        """Calls the dNet class with specific values.

        Inputs
        ---
        frId - feeder ID
        reloadDss - whether to call dNet.__init__ to reload opendss etc

        """
        self.frId = frId
        self.setCktFn()
        self.getSln(None)

        C = "cktInfo"
        fn = self.getCacheFn(C)
        if reloadDss:
            # get ckt info
            prmIn = {"yset": "Ybus"}
            dNet.__init__(
                self,
                frId,
                yBdInit=False,
                prmIn=prmIn,
                tests=[],
            )
            self.getCktInfo()
            Res = {"cktInfo": self.cktInfo}
            dNet.saveCache(fn, Res=Res, vbs=False)
        else:
            res = dNet.loadCache(fn, cache=self.caches[C], vbs=False)
            self.cktInfo = res["cktInfo"]
            self.setCktInfo()

    def getCktInfo(
        self,
    ):
        """Populate cktInfo dict attribute using cktInfoHead."""
        self.cktInfo = {val: getattr(self, val) for val in self.cktInfoHead}

    def setCktInfo(
        self,
    ):
        """Set cktInfo dict attribute to self."""
        for key, val in self.cktInfo.items():
            setattr(self, key, val)

    def setAuxCktAttributes(
        self,
    ):
        """Sets a useful derived quantities from ckts.

        Call after getLvCktInfo or after loadMlvMdl.
        """
        self.idxPlv = np.concatenate(self.ckts.LvYZsIdx)
        self.idxQlv = self.idxPlv + self.nP
        self.nPlv = len(self.idxPlv)
        self.mlvNedges = np.r_[0, np.cumsum(self.mlvNsplt)]

    @staticmethod
    def getCtype(fn, lvFrid, ldNo):
        """Get the type ('ukgds' or 'one') of a given load.

        TBH quite cumbersome at this stage - it manually goes and checks the
        filename of the corresponding circuit.
        """
        lvFrid = str(lvFrid)
        dn0 = os.path.dirname(fn)
        dn = join(dn0, "lvNetworks", "_".join(["network", lvFrid, "1", ldNo]))

        fls = os.listdir(dn)
        mn0 = "masterNetwork" + lvFrid
        ndings = ["_pruned_ukgds.dss", "_pruned_one.dss", "_pruned.dss"]
        for nding in ndings:
            if mn0 + nding in fls:
                return nding

    def getLvCktInfo(
        self,
        state=None,
        fn0=None,
    ):
        """Get LV circuit info from the current circuit.

        Assumes that there are a set of 3 phase trasformers off the
        main MV backbone. These transformers are assumed to have the name
        'transformer_...'.

        Returns a structDict self.ckts with the following properties:
        - trnIdx: the index of the transformer in d.TRN
        - lvFrid: the lv fdId base
        - ldNo: the load number on the MV circuit that is being used
        - N: the number of transformers
        - ctype: either 'ukgds' or 'one'

        - YZvIdx: the TRN bus indexes, with respect to self.YZv
        - YZmvIdx: the TRN bus indexes, with respect to self.YZv[self.mvIdx]

        - pwrs: the per-phase power through the TRN
        - V: the complex voltages on the TRN MV side
        - Vbase: the voltage base for V

        - LvYZsIdx: the YZs order for the circuit in the Full circuit
        - LvYZs: the YZs order for the circuit (with the load ID stripped)

        - LvYZvIdx: the indexes of the LV circuit buses in self.YZv
        - LvYZv: the names of the LV circuit buses (with the load ID stripped)

        - lvPrp: the string at the start of all elements in this LV ckt
        - LvYZvIdx_V: as LvYZvIdx but only output voltage indexes
        - LvYZsIdx_S: as LvYZsIdx but with P & Q indexes

        """
        cktKeys = [
            "trnIdx",
            "lvFrid",
            "ldNo",
            "YZvIdx",
            "YZmvIdx",
            "lvPrp",
            "V",
            "Vbase",
            "pwrs",
            "LvYZsIdx",
            "LvYZs",
            "LvYZvIdx",
            "LvYZv",
            "LvYZvIdx_V",
            "LvYZsIdx_S",
            "ctype",
        ]
        cc = mtDict(cktKeys)
        cc["N"] = 0

        state = self.state0 if state is None else state
        fn0 = self.fn if fn0 is None else fn0

        V0 = d.setNtwkState(state)[3:] / self.vKvbase
        i = d.TRN.First
        while i:
            if not d.TRN.Name.split("_")[0] == "transformer":
                # Basic info/updates
                ldNo = d.TRN.Name.split("_")[1][:-1]
                cc["trnIdx"].append(i)
                cc["lvFrid"].append(
                    int(d.TRN.Name.split("_")[0][2:-1]),
                )
                cc["ldNo"].append(ldNo)
                cc["ctype"].append(None)  # getCtype method has been superceded.
                cc["N"] += 1

                # Transformer indexes
                idxs = [self.YZv.index(ldNo + phs) for phs in [".1", ".2", ".3"]]
                YZvMv = vecSlc(self.YZv, self.mvIdx)
                idxsMv = [self.YZv.index(ldNo + phs) for phs in [".1", ".2", ".3"]]
                cc["YZvIdx"].append(idxs)
                cc["YZmvIdx"].append(idxsMv)

                # Circuit state
                cc["V"].append(V0[idxs])
                cc["Vbase"].append(self.vKvbase[idxs])
                cc["pwrs"].append(tp2ar(d.AE.Powers)[:3])

                # Input/output indexes
                lvPrp = "1_" + ldNo + "_"
                nPrp = len(lvPrp)
                cc["lvPrp"].append(lvPrp)
                cc["LvYZsIdx"].append(
                    np.array([i for i, yz in enumerate(self.YZs) if yz[:nPrp] == lvPrp])
                )
                cc["LvYZs"].append([yz[nPrp:] for yz in self.YZs if yz[:nPrp] == lvPrp])

                cc["LvYZvIdx"].append(
                    np.array([i for i, yz in enumerate(self.YZv) if yz[:nPrp] == lvPrp])
                )
                cc["LvYZv"].append([yz[nPrp:] for yz in self.YZ if yz[:nPrp] == lvPrp])

                cc["LvYZvIdx_V"].append(
                    np.array([i for i in cc["LvYZvIdx"][-1] if i in self.sIdx])
                )
                cc["LvYZsIdx_S"].append(
                    np.r_[cc["LvYZsIdx"][-1], cc["LvYZsIdx"][-1] + self.nP]
                )

            i = d.TRN.Next

        self.mlvIdx = np.concatenate([idxV for idxV in cc["LvYZvIdx_V"]])
        self.mlvNsplt = np.array([len(idxs) for idxs in cc["LvYZvIdx_V"]])
        self.mlvKvbase = self.vKvbase[self.mlvIdx]

        self.ckts = structDict(cc)

    def getVmlvSln(self, state):
        """Calculate mlvSln, a 'mlv-ified' version of sln.

        To see which headings go into this, see self.getSln. The main change
        is only in terms of the voltages which are returned being in the
        mlv indexes.
            - Vc, the voltages excluding the source
            - vc, V, v; derived voltages in magnitude/pu etc
            - TP, total power
            - TL, total losses
            - Cnvg, converged flag

        As with mlvCalc, returns a list of values of V for each LV circuit.
        """
        # First, recompile the circuit to the MV one if they don't match
        if self.feeder.lower() != d.DSSCircuit.Name.lower():
            self.initialiseDssCkt()

        # Then get the solution as per the normal way
        self.getSln(state)

        mlvSln = {
            "stateIn": self.sln.stateIn,
            "VsrcMeas": self.sln.VsrcMeas,
            "stateOut": self.sln.stateOut,
            "TP": self.sln.TP,
            "TL": self.sln.TL,
            "Cnvg": self.sln.Cnvg,
            "Vc": self.sln.Vc[self.mlvIdx],
            "V": self.sln.V[self.mlvIdx],
            "vc": self.sln.vc[self.mlvIdx],
            "v": self.sln.v[self.mlvIdx],
        }
        self.mlvSln = structDict(mlvSln)

    def mlvPlot(self, Vset, ax=None, pKwArgs={}, ff=None, ylm=None):
        """Utility function for plotting Vset.

        Inputs:
        - Vset: an iterable of voltages/objects
        - ax: axes to plot onto
        - pKwArgs: a dict of kwargs to go into 'plot'
        - ff: a function to transform V by (e.g. divide by voltage base)
        - ylm: y-limits (required for vlines)

        """
        if ax is None:
            fig, ax = plt.subplots()

        if ff is None:
            ff = lambda x: x

        plt.plot(ff(Vset), **pKwArgs)

        if ylm is None:
            ylm = ax.get_ylim()

        plt.vlines(
            np.r_[-0.5, np.cumsum(self.mlvNsplt) + 0.5],
            *ylm,
            linestyles="dotted",
            linewidth=0.5,
        )
        ax.set_ylim(ylm)

        ax.set_xlabel("Load index")
        ax.set_ylabel("Voltage, pu")
        plt.tight_layout()
        if self.fPrm["showFig"]:
            plt.show()

    def vAll2vLv(self, Vlv):
        """Convert a list of Vlv to a list of V in each LV circuit."""
        Vdcmp = []
        for i0, i1 in zip(self.mlvNedges[:-1], self.mlvNedges[1:]):
            Vdcmp.append(Vlv[i0:i1])

        return Vdcmp

    @staticmethod
    def fridlm2id(frId, lm):
        """Convert frid + lm to a unique ID. as f'{frId}_{lm}'."""
        return f"{frId}_{lm}"
