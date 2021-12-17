import os, sys, dss, shutil
from . import dss_utils

import os, sys, pickle, zipfile, calendar
from datetime import datetime, timedelta
from pprint import pprint
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from timeit import default_timer as timer
from collections import OrderedDict as odict
from copy import deepcopy
import logging
from pathlib import Path
from bunch import Bunch
from progress.bar import Bar

from .funcsPython_turing import (
    fillplot,
    get_path_files,
    get_path_dirs,
    sff,
    listT,
    mtOdict,
    dds,
    csvIn,
    mtDict,
    rngSeed,
    o2o,
)
from . import funcsDss_turing
from .funcsMath_turing import vecSlc, tp2ar
from . import slesNtwk_turing as snk

from . import azureOptsXmpls as aox

from io import BytesIO
from azure.storage.blob import BlobServiceClient
import gc
from .config import get_settings


data_dir = os.path.join(Path(__file__).parent, "data")
# Load opendss

# dssObj = loadDss()
# d = funcsDss_turing.dssIfc(dssObj)
d = funcsDss_turing.dssIfc(dss.DSS)

# Based on trial and error, loosely based on I&C profiles, pp25 of T. Short
xx = np.linspace(0, 1, 144)
pi2 = np.pi * 2
ps = pi2 / 10
ic00_prof = 0.8 + 0.15 * (
    np.cos(-xx * pi2 - pi2 / 2 + ps)
    + 0.2 * np.cos(-xx * pi2 * 2 - 0.15 * pi2 + ps)
    + 0.1 * np.cos(-xx * pi2 * 3 - 0.15 * pi2 + ps)
)


def xtnd_ts(
    xx,
    n,
    opt="interp",
):
    """Extend the (timeseries) matrix xx by n.

    Note that 'interp' uses a *periodic* interpolation to extend at the end
    of a day.

    Inputs
    ---
    xx - the data
    n - the number of periods to upsample to.
    opt - the mode to run in, either 'kron' for kron extension, or 'interp'.
    """
    if opt == "kron":
        # Simply return the data extended out.
        return np.kron(xx, np.ones(n))
    elif opt == "interp":
        # Return the data, linearly interpolated.
        nx = len(xx) if xx.ndim == 1 else xx.shape[1]
        t0 = np.arange(0, nx, 1)
        t1 = np.arange(0, nx, 1 / n)
        if xx.ndim == 1:
            return np.interp(
                t1,
                t0,
                xx,
                period=nx,
            )
        else:
            return np.array(
                [
                    np.interp(
                        t1,
                        t0,
                        x,
                        period=nx,
                    )
                    for x in xx
                ]
            )


class turingNet(snk.dNet):
    """A MLV class for development of code alongside V2G with Turing."""

    str_lvmv = [
        "lv",
        "mv",
    ]

    # Peak to houses converted assuming 1.3 kVA. See
    # "Hybrid European MV–LV Network Models for Smart Distribution Network
    # Modelling", M. Deakin et al (Powertech 2021) for details of all
    # assumptions. On arxiv: https://arxiv.org/abs/2009.14240v1
    s_to_nhouses = 1.3  # kVA

    max_lv_ckts = 10

    rd0 = {
        "rand_seed": 0,
    }

    def __init__(
        self,
        frId=40,
        frId0=None,
        rundict=rd0,
        mod_dir=os.path.join(os.getcwd(), "_network_mod"),
    ):
        """Initialise - first calling snk.dNet, then snk.mlvNet for more info.

        Inputs
        ---
        frId - the feeder ID
        frId0 - the modified feeder ID (when calling modified MVLV networks)
        rundict - the 'rundict' options dictionary (see azureOptsXmpls)

        """

        snk.dNet.__init__(
            self,
            frId=frId,
            tests=[],
            yBdInit=False,
            prmIn={"nomLdLm": 1.0, "yset": None},
            mod_dir=mod_dir,
        )

        fn0 = None if frId0 is None else os.path.join(mod_dir, "_network_mod")
        self.frId0 = frId0
        snk.mlvNet.getLvCktInfo(self, fn0=os.path.join(fn0, "lvNetworks"))
        # snk.mlvNet.setAuxCktAttributes(self,) # Not too sure if needed?

        # Set to 'self' all attributes of rundict not in their own dicts
        self.rd = {"readme": "rd - as rundict's simple key-value pairs."}
        for k, v in rundict.items():
            if not type(v) is dict:
                self.rd[k] = v

        # Set up a main random number generator
        self.rng = rngSeed(self.rd["rand_seed"])
        self.set_base_tags()
        self.set_ckts_ctype()
        assert (
            self.ckts.N <= self.max_lv_ckts
        ), f"Please choose fewer than {self.max_lv_ckts} LV circuits."

        if "dmnd_gen_data" in rundict.keys():
            self.set_ldsi(
                dgd=rundict["dmnd_gen_data"],
            )

            # Set the profiles self.p and then the demand self.dmnd
            self.set_prfs(
                tsp=rundict["simulation_data"]["ts_profiles"],
            )
            self.set_dmnd(
                tsp=rundict["simulation_data"]["ts_profiles"],
            )

            # Set circuit power limits
            self.set_powers()
        else:
            print(
                "--> No dmnd_gen_data, and so not setting self.ldsi, self.p or self.dmnd."
            )

    def set_powers(
        self,
    ):
        """Set the circuit power limits for the branches (in self.pmryLnsi).

        Branch limits taken from:
        https://github.com/sedg/ukgds/blob/master/HV_UG-OHa.xls ;
        https://github.com/sedg/ukgds/blob/master/HV_UG.xls
        """
        if self.frId0 == 1060:
            # Table 4 with NPG Page 19
            ckts2rr = {
                1: 6.82,
                3: 6.82,
                4: 8.86,
                7: 8.86,
                8: 8.86,
            }
            ckts = [1, 1, 1, 3, 4, 4, 7, 8]
            rr = np.array([ckts2rr[i] for i in ckts])
        elif self.frId0 == 1061:
            rr = np.ones(3) * 8.86
        else:
            rr = np.nan * np.zeros(len(self.pmryLnsi))

        b2 = np.array(d.getObjAttr(d.LNS, "Bus2"))[self.pmryLnsi]
        self.fdr2pwr = odict({b: p for b, p in zip(b2, rr)})

    def set_base_tags(
        self,
    ):
        """Finds the nominal indices used with record_solution.

        Sets
        ---
        self.mvlvi - indexes of MV and LV loads in d.LDS
        self.xfmri - indexes of primary and secondary trns in d.TRN
        self.pmryLnsi - indexes of lines conn to 1mry sub in d.LNS
        self.sdryLnsi - indexes of lines conn to 2dry sub in d.LNS
        """
        # First splite loads into LV (<1 kV) and MV (otherwise)
        kVkW = listT([d.getObjAttr(d.LDS, "kV"), d.getObjAttr(d.LDS, "kW")])
        self.mvlvi = Bunch(
            {
                "mv": [i for i, (v, p) in enumerate(kVkW) if v > 1.0 and p != 0],
                "lv": [i for i, (v, p) in enumerate(kVkW) if v <= 1.0],
            }
        )
        self.mvlvi.nmv = len(self.mvlvi.mv)
        self.mvlvi.nlv = len(self.mvlvi.lv)

        # Create an OrderedDict for the residential loads split by ckt
        ldn = d.getObjAttr(d.LDS, "Name")
        self.mvlvi.lv_n = mtOdict(self.ckts.ldNo)
        for i, n in enumerate(ldn):
            if n.count("_") == 3:
                self.mvlvi.lv_n[n.split("_")[1]].append(i)

        # Similarly find the transformer voltages
        xnms = d.getObjAttr(d.TRN, "Name")
        self.xfmri = Bunch(
            {
                "pmry": [i for i, n in enumerate(xnms) if "tra" in n],
                "sdry": [i for i, n in enumerate(xnms) if not "tra" in n],
            }
        )
        self.xfmri.npmry = len(self.xfmri.pmry)
        self.xfmri.nsdry = len(self.xfmri.sdry)

        # Get the lines connected to the primary substation
        d.TRN.First
        pmryBus = d.AE.BusNames[1]
        self.pmryLnsi = [
            i for i, b in enumerate(d.getObjAttr(d.LNS, "Bus1")) if b == pmryBus
        ]

        # Then get the buses of all secondary LV substations
        trnBuses = d.getObjAttr(d.TRN, val=None, AEval="BusNames")
        sdryBus = [trnBuses[i][1] for i in self.xfmri.sdry]

        # ...and subsequently the indexes of lines that connect to them
        self.sdryLnsi = mtOdict(self.ckts.ldNo)
        for i, lnb in enumerate(d.getObjAttr(d.LNS, "Bus1")):
            if lnb in sdryBus:
                self.sdryLnsi[lnb.split("_")[1]].append(i)

    def set_ckts_ctype(
        self,
    ):
        """New version of the function mlvNet.getCtype, using self.sdryLnsi.

        Sets self.ckts.ctype to be the number of feeders per LV network.
        """
        for i, ldNo in enumerate(self.ckts.ldNo):
            self.ckts.ctype[i] = len(self.sdryLnsi[ldNo])

    ldsi_opts_readme = {
        "None": "Do not allocate (can be set to all types)",
        "lv": "All LV loads",
        "mv": "Allocate to all MV loads",
        "slr_pen": "Allocates solar PV to loads randomly, with penetration slr_pen",
        "hps_pen": "As slr_pen; allocates HPs randomly, with pen. hps_pen",
        "ev_pen": "As slr_pen; allocates HPs randomly, with pen. ev_pen",
        "res_pen": "As slr_pen; allocates residential loads to MV with pen res_pen",
        "odds": "Allocate to every other load (odd nos).",
        "evens": 'Complement of "odds".',
        "rs": "Allocate to all residential loads. Equivalent to *_pen=100. Used only with slr, ovnt, dytm, hps.",
        "not_rs": 'Complemenet of "rs" - used for MV IC loads, for example.',
        "ic": 'As with "rs", but uses only residential loads [is therefore only MV].',
        "List": "A list of bus numbers (as strings) indicating the position of lumped DGs, FCSs. (Powers [in kW] specified in ts_profiles.dgs.mv, ts_profiles.fcs.mv)",
    }

    ldsi_opts_dict = {
        "rs": {
            "lv": [
                "lv",
            ],
            "mv": [
                "mv",
                "odds",
                "evens",
                "rs_pen",
            ],
        },
        "ic": {
            "lv": [],  # lumped I&C only
            "mv": [
                "mv",
                "odds",
                "evens",
                "not_rs",
            ],
        },
        "ovnt": {
            "lv": [
                "rs",
                "ev_pen",
            ],
            "mv": [
                "rs",
                "ic",
            ],
        },
        "dytm": {
            "lv": [
                "rs",
            ],
            "mv": [
                "rs",
                "ic",
            ],
        },
        "slr": {
            "lv": [
                "lv",
                "slr_pen",
            ],
            "mv": [
                "mv",
                "rs",
            ],
        },
        "hps": {
            "lv": [
                "lv",
                "hps_pen",
            ],
            "mv": [
                "mv",
                "rs",
            ],
        },
        "dgs": {
            "lv": [],  # Lumped DGs only
            "mv": [
                list,
            ],
        },
        "fcs": {
            "lv": [],  # Lumped DGs only
            "mv": [
                list,
            ],
        },
    }

    def set_ldsi(
        self,
        dgd={},
    ):
        """Set ldsi, the indexes for each demand type using dmnd_gen_data (dgd).

        Flags which are set (via a list of indexes in d.LDS):
        - residential (LV, MV lumped): rs;
        - I&C (LV, MV lumped): ic;
        - daytime / overnight charging: dytm / ovnt;
        - solar PV: slr;
        - heat pump: hps;
        - lumped DGs: dgs;
        - Fast charging stations: fcs.

        To see the options that each of these types can be set to, see the
        dictionary pprint(self.ldsi_opts_dict) and their corresponding
        descriptions pprint(self.ldsi_opts_readme).
        """
        # Initialise and set all values with None to be empty.
        self.ldsi = Bunch({})
        _ = [setattr(self.ldsi, k, Bunch({"lv": [], "mv": []})) for k in dgd.keys()]
        self.ldsi.kw = list(dgd.keys())

        # Utility funcs for choosing indexes
        choose_n = lambda xx, n: self.rng.choice(
            xx,
            size=n,
            replace=False,
        )
        pen2n = lambda pen, vv: int(np.round(self.mvlvi["n" + vv] * pen / 100))
        nms = d.getObjAttr(
            d.LDS,
            AEval="BusNames",
        )

        # Functions to return indices into self.ldsi
        f_vv = lambda vv: deepcopy(self.mvlvi[vv])  # 'lv', 'mv'
        f_xx = lambda xx: deepcopy(xx)  # 'rs', 'ic'
        f_nms = lambda buses: [nms.index([nn]) for nn in buses]

        def f_pen(
            kk,
            rs_vv,
            vv,
        ):
            nn = pen2n(
                self.rd[kk],
                vv,
            )  # kk as 'ev_pen', 'hps_pen', etc
            return choose_n(
                rs_vv,
                nn,
            )

        def f_oe(
            vv,
            oe,
        ):  # 'odds', 'evens'
            assert vv == "mv"
            offset = 0 if oe == "evens" else 1
            return f_vv(
                vv,
            )[offset::2]

        def f_not_rs(vv):
            assert vv == "mv"
            return [i for i in self.mvlvi[vv] if i not in self.ldsi["rs"][vv]]

        # Now loop through all load types and voltages, settings ldsi indexes.
        check_val = lambda ld, vv: dgd[ld][vv] in self.ldsi_opts_dict[ld][vv]
        for vv in self.str_lvmv:
            # First set residential & IC loads:
            for ld in [
                "rs",
            ]:
                if check_val(ld, vv):
                    if dgd[ld][vv] in [
                        "lv",
                        "mv",
                    ]:
                        self.ldsi[ld][vv] = f_vv(
                            vv,
                        )
                    elif dgd[ld][vv] in ["odds", "evens"]:
                        self.ldsi[ld][vv] = f_oe(
                            vv,
                            dgd[ld][vv],
                        )
                    elif dgd[ld][vv] in [
                        "rs_pen",
                    ]:
                        self.ldsi[ld][vv] = f_pen(
                            "rs_pen",
                            self.mvlvi[vv],
                            vv,
                        )
                else:
                    assert dgd[ld][vv] is None

            for ld in [
                "ic",
            ]:
                if check_val(ld, vv):
                    if dgd[ld][vv] in [
                        "lv",
                        "mv",
                    ]:
                        self.ldsi[ld][vv] = f_vv(
                            vv,
                        )
                    elif dgd[ld][vv] in ["odds", "evens"]:
                        self.ldsi[ld][vv] = f_oe(
                            vv,
                            dgd[ld][vv],
                        )
                    elif dgd[ld][vv] in [
                        "not_rs",
                    ]:
                        self.ldsi[ld][vv] = f_not_rs(vv)
                else:
                    assert dgd[ld][vv] is None

            # Solar / HP / EV profiles that are dependent on IC, RS locs:
            for ld in [
                "dytm",
                "ovnt",
                "slr",
                "hps",
            ]:
                if check_val(ld, vv):
                    kk = dgd[ld][vv]
                    if dgd[ld][vv] in [
                        "lv",
                        "mv",
                    ]:
                        self.ldsi[ld][vv] = f_vv(
                            vv,
                        )
                    elif kk in [
                        "ic",
                        "rs",
                    ]:
                        self.ldsi[ld][vv] = f_xx(self.ldsi[kk][vv])
                    elif kk in [
                        "ev_pen",
                        "hps_pen",
                        "slr_pen",
                    ]:
                        self.ldsi[ld][vv] = f_pen(
                            kk,
                            self.ldsi.rs[vv],
                            vv,
                        )
                else:
                    assert dgd[ld][vv] is None

            # Large DGs & FCSs are located only at MV buses.
            for ld in ["dgs", "fcs"]:
                if type(dgd[ld][vv]) is list:
                    assert vv == "mv"
                    self.ldsi[ld][vv] = f_nms(dgd[ld][vv])

            # Then set all values of 'n*'
            _ = [
                setattr(self.ldsi[k], "n" + vv, len(self.ldsi[k][vv]))
                for k in list(self.ldsi.kw)
            ]

    def plotXvNetwork(
        self,
        xv="mv",
        pType=None,
        pnkw={},
        net=None,
    ):
        """Plot the XV network (ie MV or LV). Based on plotNetwork from dNet.

        Difference: we will not plot the whole network, ONLY a subnetwork.

        Inputs
        ---
        xv - either 'mv', 'lv', or 'lvraw'
        pType - the plot type
        pnkw - plotNetwork kwargs arguments. Also has a few bonuses:
            - 'txtOpts' option to indicat to plot the bus numbers
            - 'lvnFlag' option to highlight LV Network buses
            - 'dgFlag' option to highlight MV DG (dgs) buses + FCS buses.
            - 'txtFs' fontsize for plotting txtOpts with
        net - an LV network ID
        """
        self.setupXvPlots(
            xv,
            net=net,
        )
        txtOpts = pnkw.get("txtOpts", None)
        lvnFlag = pnkw.get(
            "lvnFlag",
            True,
        )
        dgFlag = pnkw.get(
            "dgFlag",
            False,
        )  # also does fcs
        txtFs = pnkw.get(
            "txtFs",
            10,
        )
        figsize = pnkw.get(
            "figsize",
            (13, 7),
        )

        lvn_clr = "r"
        dg_clr = "b"
        fcs_clr = "g"
        lvn_dict = {} if not pType == "B" else {"facecolor": lvn_clr, "alpha": 0.3}
        dg_dict = {} if not pType == "B" else {"facecolor": dg_clr, "alpha": 0.3}
        fcs_dict = {} if not pType == "B" else {"facecolor": fcs_clr, "alpha": 0.3}

        # First get the lumped MV loads and LV network loads if wanted
        if (
            pType
            in [
                "B",
                "s",
                "p",
                "q",
            ]
            or not txtOpts is None
        ):
            lvntwkopts = {
                "color": lvn_clr,
                "marker": ".",
                "ms": 10,
                "mew": 1,
                "ls": "",
                "zorder": 30,
                "mfc": "None",
            }
            dgntwkopts, fcsntwkopts = [deepcopy(lvntwkopts) for i in range(2)]
            dgntwkopts.update({"color": dg_clr})
            fcsntwkopts.update({"color": fcs_clr})

            lvNtwkBuses = self.ckts.ldNo
            icMvBuses, rsMvBuses, dgBuses, fcsBuses = [
                self.ldsi2buses(self.ldsi[k].mv) for k in ["ic", "rs", "dgs", "fcs"]
            ]

            aln = ["top", "bottom"] * (d.LDS.Count // 2)
            fig, ax = plt.subplots(
                figsize=figsize,
            )
            pnkw.update({"ax": ax})

        # Then plot any buses
        if txtOpts == "all":
            # Create the plot and mark bus numbers
            self.plotLoadNos(
                icMvBuses,
                ax=ax,
                aln=aln,
                fontsize=txtFs,
            )
            self.plotLoadNos(
                rsMvBuses,
                ax=ax,
                aln=aln,
                fontsize=txtFs,
            )
            if len(lvn_dict) == 0:
                self.plotLoadNos(
                    lvNtwkBuses,
                    ax=ax,
                    aln=aln,
                    fontsize=txtFs,
                )
            else:
                self.plotLoadNos(
                    lvNtwkBuses, ax=ax, aln=aln, fontsize=txtFs, bbox=lvn_dict
                )
            if dgFlag:

                def bus2alni(k):
                    try:
                        return aln[icMvBuses.index(k)]
                    except:
                        return aln[rsMvBuses.index(k)]

                dg_aln, fcs_aln = [
                    [bus2alni(k) for k in bb] for bb in [dgBuses, fcsBuses]
                ]

                self.plotLoadNos(
                    dgBuses,
                    ax=ax,
                    aln=dg_aln,
                    fontsize=txtFs,
                    bbox=dg_dict,
                    alpha=0,
                )
                self.plotLoadNos(
                    fcsBuses,
                    ax=ax,
                    aln=fcs_aln,
                    fontsize=txtFs,
                    bbox=fcs_dict,
                    alpha=0,
                )

        if pType == "B":
            # Plot the 'build' of the network, showing detailed LV vs
            # lumped Res vs lumped I&C

            # Create a 'scores' idct to pass to plotNetwork, plus colors
            scrs = {k: 0 for i, k in enumerate(rsMvBuses + lvNtwkBuses)}
            scrs.update({k: 1 for k in icMvBuses})

            mnmx = [-1, 1.2]
            c_lv, c_mv = [cm.viridis((i - mnmx[0]) / np.diff(mnmx)[0]) for i in [0, 1]]

            # Plot the LV network 'flag'
            if lvnFlag:
                self.plotNtwkLocs(ax, lvn_clr, buses=self.ckts.ldNo)

            # Plot the DGs and FCSs
            if dgFlag:
                _ = [
                    self.plotNtwkLocs(ax, clr, idxs=self.ldsi[yy].mv)
                    for clr, yy in zip([dg_clr, fcs_clr], ["dgs", "fcs"])
                ]

            # Do the legend (nb ax_scat a bit messy for legends)
            (plt_lv,) = plt.plot(
                np.nan,
                np.nan,
                ".",
                color=c_lv,
                mec="k",
                mew=0.3,
                ms=10,
            )
            (plt_mv,) = plt.plot(
                np.nan,
                np.nan,
                ".",
                color=c_mv,
                mec="k",
                mew=0.3,
                ms=10,
            )
            (xLg,) = plt.plot(np.nan, np.nan, **lvntwkopts)
            lbls0 = [
                "Res. demand,\nlumped",
                "I&C demand,\nlumped",
                "LV Modelled",
            ]
            if dgFlag:
                (xDg,) = plt.plot(np.nan, np.nan, **dgntwkopts)
                (xFcs,) = plt.plot(np.nan, np.nan, **fcsntwkopts)
                plt.legend(
                    [plt_lv, plt_mv, xLg, xDg, xFcs],
                    lbls0 + ["Large DG", "EV Fast Chrg. Stn."],
                )
            else:
                plt.legend(
                    [plt_lv, plt_mv, (plt_lv, xLg)],
                    lbls0,
                    loc=[1.02, 0.2],
                )

            # Create plotting options to send to plotNetwork
            pnkw.update(
                {
                    "minMax0": mnmx,
                    "score0": np.array([scrs.get(b, np.nan) for b in self.bus0v]),
                }
            )

        if (
            pType
            in [
                "s",
                "p",
                "q",
            ]
            and xv == "mv"
        ):
            # When we are looking at loads in the MV circuits of MV/LV, we also
            # consider lumped LV circuits as load on MV.
            mvBuses = self.ldsi2buses(self.mvlvi.mv)  # lumped only
            lvmvBuses = self.ckts.ldNo

            mvPwrs_S = np.array(d.getObjAttr(d.LDS, "kW",)) + 1j * np.array(
                d.getObjAttr(
                    d.LDS,
                    "kvar",
                )
            )
            lvmvPwrs_S = self.ckts.pwrs

            # Select the right data and scores.
            fsel = {"s": np.abs, "p": np.real, "q": np.imag}
            mvPwrsSel = [fsel[pType](xx) for xx in mvPwrs_S[self.mvlvi.mv]]
            lvmvPwrsSel = [fsel[pType](sum(xx)) for xx in lvmvPwrs_S]

            scrs = {k: v for k, v in zip(mvBuses + lvmvBuses, mvPwrsSel + lvmvPwrsSel)}
            cbt0s = {
                "p": "Load, kW",
                "q": "Reactive pwr, kVAr",
                "s": "Apparent power, kVA",
            }

            pnkw.update(
                {
                    "cmapSet": self.fPrm_["cms"].get(pType, plt.cm.viridis),
                    "cbTtl": cbt0s[pType],
                    "score0": np.array([scrs.get(b, np.nan) for b in self.bus0v]),
                }
            )

            # Plot the LV network locations
            if lvnFlag:
                self.plotNtwkLocs(ax, lvn_clr, buses=self.ckts.ldNo)

            # Update so that only the voltage locations are plotted
            pType = "v"

        self.plotNetwork(pType, **pnkw)

    def plotNtwkLocs(
        self,
        ax,
        clr,
        buses=None,
        idxs=None,
    ):
        """Plot specific network locations, either buses or indexes, with clr.

        Either pass in buses xor idxs.
        """
        assert (buses is None) != (idxs is None)  # != as xor

        pdict = {
            "s": self.fPrm_["pms"],
            "facecolors": "None",
            "marker": ".",
            "zorder": 30,
            "edgecolors": clr,
            "linewidth": 1.2,
        }

        if buses is None:
            buses = self.ldsi2buses(idxs)

        if len(buses) == 0:
            return None

        print(buses)
        for bus in buses:
            ax_scat = ax.scatter(*self.busCoords[bus], **pdict)

        return ax_scat

    def ldsi2buses(self, ldsi_idx):
        """Convert ldsi_idx indexes to bus names for use with plotting funcs."""
        return [
            b[0].split("_")[-1]
            for b in vecSlc(d.getObjAttr(d.LDS, AEval="BusNames"), ldsi_idx)
        ]

    def plotLoadNos(self, buses, ax=None, aln=None, **kwargs):
        """Plot load numbers and locations of buses.

        Inputs
        ---
        idxs - the indexes in d.LDS (if None plot all);
        ax - the axis to plot onto (if None create a new axis);
        aln - a list of vertical alignments
        kwargs - kwargs to pass into ax.text
        """
        aln = ["bottom" for i in range(len(idxs))] if aln is None else aln
        if ax is None:
            fig, ax = plt.subplots()

        for ii, bus in enumerate(buses):
            txt = ax.text(
                *self.busCoords[bus],
                bus,
                verticalalignment=aln[ii],
                **kwargs,
            )

        return ax

    def setupXvPlots(
        self,
        xv,
        net=None,
    ):
        """Get data required for using plotNetwork via plotXvNetwork.

        Based on self.setupPlots; see self.plotXvNetwork for more info
        """
        # regBuses is as-in setupPlots
        self.regBuses = d.getRgcNds(self.YZvIdxs)[0]
        self.srcReg = 0  # for arguments sake
        self.branches = d.getBranchBuses()

        self.busCoords = {k: (np.nan, np.nan) for k in d.DSSCircuit.AllBusNames}
        dn0 = os.path.join(data_dir, "coords")
        if xv == "mv":
            m0 = d.DSSCircuit.Name.split("_")[1].upper()
            m0 = m0[:-1] + (m0[-1].lower() if m0[-1] in ["A", "B"] else m0[-1])
            fn = os.path.join(dn0, f"HV_{m0}_buscoords.csv")

            # Load the data and create buscoords.
            data = csvIn(fn, hh=0)
            for (k, x, y) in data:
                self.busCoords[k] = (float(x), float(y))
        elif xv in [
            "lv",
            "lvraw",
        ]:
            # lv a little more tricky. First find the lv network & feeders
            lnsNames = d.getObjAttr(d.LNS, "Name")
            feeders = [n[-1] for n in vecSlc(lnsNames, self.sdryLnsi[net])]
            lvn = self.ckts.lvFrid[self.ckts.ldNo.index(net)]

            # Load each of the LV feeders
            bdata = []
            lraw = []
            for ff in feeders:
                dn = os.path.join(dn0, "lvns_coords", f"network_{lvn}", f"Feeder_{ff}")
                bdata.extend(csvIn(os.path.join(dn, "XY_Position.csv"), hh=0))
                lraw.extend(csvIn(os.path.join(dn, "LinesUnq.txt"), hh=0))

            # Change the names of each of the bus names as appropriate
            bclv = {f"1_{net}_{k}": (float(x), float(y)) for (k, x, y) in bdata}
            if xv == "lvraw":
                for k in self.busCoords.keys():
                    self.busCoords[k] = bclv.get(k, (np.nan, np.nan))
            else:
                self.busCoords.update(bclv)

            # If tidy LV, update the branches
            if xv == "lv":
                cmpds = [k for k in self.branches.keys() if "_cmpd_" in k]
                _ = [self.branches.__delitem__(k) for k in cmpds]

                # Then add all of the branches from the full networks.
                for l in lraw:
                    lid = l[0][13:].split(" ")[0]
                    name = "_".join(["Line.line1", net, lid])
                    buses = [
                        l[0].split(f"Bus{i}=")[1].split(" ")[0] for i in range(1, 3)
                    ]
                    self.branches[name] = [f"1_{net}_{b}" for b in buses]

            # Set the source bus as the head of the first feeder
            self.busCoords[self.vSrcBus] = bclv[f"1_{net}_{bdata[0][0]}"]

        # Make sure branches with one coord can be drawn
        for [b0, b1] in self.branches.values():
            bc0, bc1 = [self.busCoords[b][0] for b in [b0, b1]]
            if np.isnan(bc0) and not np.isnan(bc1):
                self.busCoords[b0] = self.busCoords[b1]
            elif np.isnan(bc1) and not np.isnan(bc0):
                self.busCoords[b1] = self.busCoords[b0]

        # Finally get the voltage and power lists
        self.getBusPhs()

    def set_prfs(
        self,
        tsp,
    ):
        """Set the profiles Bunch self.p or 'raw' temporal profiles.

        The rows of each profile correspond to different loads, whilst the
        columns correspond to different time instances.

        Each of the profiles are assumed to correspond to one day's operation,
        and the scale is EITHER in kW or normalised.

        At the end of this function we also collapse any profiles with multiple
        values to a single mean, called "*_", for use when setting the profiles.
        """
        self.p = Bunch()

        assert (144 % tsp["n"]) == 0

        # Load the CREST model
        fndemand = os.path.join(data_dir, "ciredData.pkl")
        with open(fndemand, "rb") as file:
            crestData = pickle.load(file)

        ddsn = len(crestData[1]["peNet"]) // tsp["n"]  # downsample no
        self.p.crest = dds(crestData[1]["peNet"], ddsn).T / 1e3  # W to kW

        # Similarly load the Element Energy data;
        self.p.rsev, self.p.icev = self.load_ee_data(
            tsp["n"],
        )
        assert len(self.p.rsev) == tsp["n"]
        assert len(self.p.icev) == tsp["n"]

        # Then the NREL solar data
        self.p.solar0 = self.load_solar0(
            tsp["n"],
        )

        # Load the USB fast charging station profiles
        self.p.usb_fcs = self.load_usb_fcs_data(
            tsp["n"],
        )

        # Set the ic00 demand
        self.p.ic00 = dds(ic00_prof, 144 // tsp["n"])

        # Load the four heat pump profiles
        hp_profiles = self.load_hp_profiles(
            nn=tsp["n"],
        )
        _ = [setattr(self.p, "hp_" + k, v) for k, v in hp_profiles.items()]

        # Load the heat pump profiles from Love et al
        hp_profiles_love = self.load_hp_love_profiles(
            nn=tsp["n"],
        )
        _ = [
            setattr(self.p, "hp_love_" + calendar.month_abbr[int(k[-1])], v)
            for k, v in hp_profiles_love.items()
        ]

        # Load the EV_urbanprofiles data
        self.p.ev_urban = self.load_ev_urbanprofiles(tsp["n"])

        # Load the EV ACN data
        self.p.ev_acn = self.load_ev_acn(tsp["n"])

        # Laod the Electric Nation EV data
        self.p.ev_encc = self.load_ev_encc(tsp["n"])

        # Load the smart meter data
        self.p.uss24_urban = self.load_uss24_urban(tsp["n"])

        # Load the csv_in data:
        self.p.csv_in = self.load_csv_in(tsp["n"])

        # Collapse 2-dimensional profiles to a mean and append.
        means = {
            k: np.mean(
                v,
                axis=0,
            )
            for k, v in self.p.items()
            if v.ndim == 2
        }
        _ = [setattr(self.p, k + "_", v) for k, v in means.items()]

        # Check that the dimensions are all equal
        assert len(set([len(x.T) for x in self.p.values()])) == 1

    # Explanation of all options in dmnd_opts_dict
    dmnd_opts_readme = {
        "crest": "a set of generic profiles created by the CREST tool from Loughborough univeristy.",
        "uss24_urban": "smart meter data from CLNR",
        "crest_": "See crest.",
        "uss24_urban_": "See uss24_urban",
        "ic00": "An I&C profile loosely based on I&C profiles from the T. Short's 'Electric Power Distribution Handbook', 2014, Ch 1.",
        "rsev": "estimated profile from Element Energy report (data in directory ./data/ev-profile-data),",
        "ev_urban": "assign profiles from ev_urban dataset",
        "ev_acn": "EV workplace charging from Caltech for two weeks worth of weekdays. Data from https://ev.caltech.edu/dataset.",
        "ev_encc": 'Electric Nation "Crowd Charge" data from 2018 for one weeks worth of weekdays, for both 3.6 and 7 kW chargers.',
        "usb_fcs": "fast charging station data from the USB, Newcastle Helix",
        "csv_in": "assign according to csv_in",
        "icev": "See rsev.",
        "ev_urban_": "See ev_urban",
        "csv_in_": "See csv_in",
        "solar0": "a nominal profile based on a (>100 MW) solar plant in New York, USA, scaled down to 2 kW panels.",
        "[_profile_,[pwrA, pwrB, ...]]": """[dgs.mv, fcs.mv]
        use _profile_ for the profile,
        but use pwr0, pwr1, ... for kW of each DG/FCS. Is intend for use
        only when there are a small number of powers (e.g. when selecting
        individual, large DGs or FCSs on given feeders)
        _profile_: 
        - 'solar0', as slr. [DGs],
        - 'uss24_urban_', See uss24_urban. [FCSs]
        """,
        "hp_mid_weekday": "Profiles based on the profiles of J. Love et al.",
        "hp_mid_weekend": "See hp_mid_weekday.",
        "hp_weekday": "See hp_mid_weekday.",
        "hp_weekend": "See hp_mid_weekday.",
    }

    # The big dict of options!
    dmnd_opts_dict = {
        "rs": {
            "lv": [
                "crest",
                "uss24_urban",
            ],
            "mv": [
                "crest_",
                "uss24_urban_",
            ],
        },
        "ic": {
            "lv": [],
            "mv": [
                "ic00",
            ],
        },
        "ovnt": {
            "lv": [
                "rsev",
                "ev_urban",
                "csv_in",
                "ev_encc",
            ],
            "mv": [
                "rsev",
                "ev_urban_",
                "csv_in_",
                "ev_encc_",
            ],
        },
        "dytm": {
            "lv": [],
            "mv": [
                "icev",
                "ev_urban_",
                "csv_in_",
                "ev_acn_",
            ],
        },
        "slr": {
            "lv": [
                "solar0",
            ],
            "mv": [
                "solar0",
            ],
        },
        "dgs": {
            "lv": [],
            "mv": [
                None,
            ],  # None for [_profile_,[pwrA, pwrB, ...]]
        },
        "fcs": {
            "lv": [],
            "mv": [
                "usb_fcs",
                None,
            ],  # usb_fcs as data from USB; None as dgs mv
        },
        "hps": {
            "lv": [pp for pp in dmnd_opts_readme.keys() if "hp" in pp],
            "mv": [pp for pp in dmnd_opts_readme.keys() if "hp" in pp],
        },
    }

    def set_dmnd(
        self,
        tsp,
    ):
        """Load the underlying demand curves for all specified loads.

        Types set are specifed in self.ldsi.kw.

        Each is split into either a 'lumped' (mv) or 'individual' (lv) load.

        Each profile is indexed by an index i, with j the hourly demand. The
        units of ALL of the values are in kW. Create the loads matrix using
        self.get_lds_kva.

        ***NB***: this function is sometimes dependent on the the current state
                  of d.LDS (which changes if opendss is changed.


        Inputs
        ---
        tsp: ts_profiles (time series profiles dict) from runDict.

        Sets
        ---
        self.dmnd, a bunch-of-bunches for each of the load types in self.ldsi.kw

        Options in tsp
        ---
        All can be set to None, which assigns the profiles zeros.

        Use pprint(self.dmnd_opts_readme) for a description of the different
        profiles, and pprint(self.dmnd_opts_dict) for the possible combinations
        that can beused with each load type.

        """
        # Initialise all as zeros of an appropriate dimension
        self.dmnd = Bunch(
            {
                k: Bunch(
                    {
                        vv: np.zeros((self.ldsi[k]["n" + vv], tsp["n"]))
                        for vv in self.str_lvmv
                    }
                )
                for k in self.ldsi.kw
            }
        )

        # Load the current kVA ratings of existing loads for scaling
        lds0 = np.array(
            d.getObjAttr(
                d.LDS,
                "kva",
            )
        )
        nhses = lds0 / self.s_to_nhouses  # no houses per lds0

        # Utility func - check option and non-zero no of loads for given voltage
        check_val = (
            lambda ld, vv: tsp[ld][vv] in self.dmnd_opts_dict[ld][vv]
            and self.ldsi[ld]["n" + vv]
        )

        # Residential demand rs
        if check_val("rs", "lv"):
            pp = self.p[tsp["rs"]["lv"]]
            assert pp.ndim == 2
            self.dmnd.rs.lv = np.array(
                [pp[i % pp.shape[0]] for i in range(self.ldsi.rs.nlv)]
            )

        if check_val("rs", "mv"):
            # Assign the mean profile to all loads
            pp = self.p[tsp["rs"]["mv"]]
            self.dmnd.rs.mv = np.array([pp * nhses[i] for i in self.ldsi.rs.mv])

        # I&C demand ic
        pass  # ic lv demand

        if check_val("ic", "mv"):
            # Assign the I&C profile
            pp = self.p[tsp["ic"]["mv"]]
            ic00 = pp / max(pp)
            self.dmnd.ic.mv = np.array([lds0[i] * ic00 for i in self.ldsi.ic.mv])

        # Set the overnight charging profiles
        if check_val("ovnt", "lv"):
            pp = self.p[tsp["ovnt"]["lv"]]
            if pp.ndim == 1:
                self.dmnd.ovnt.lv = np.array([pp for i in range(self.ldsi.ovnt.nlv)])
            else:
                self.dmnd.ovnt.lv = np.array(
                    [pp[i % pp.shape[0]] for i in range(self.ldsi.ovnt.nlv)]
                )

        # overnight mv charging
        if check_val("ovnt", "mv"):
            pp = self.p[tsp["ovnt"]["mv"]]
            frac = self.ldsi.ovnt.nlv / self.mvlvi.nlv
            self.dmnd.ovnt.mv = np.array(
                [frac * nhses[i] * pp for i in range(self.ldsi.ovnt.nmv)]
            )

        # Set the daytime charging profiles
        pass  # daytime lv charging

        if check_val("dytm", "mv"):
            pp = self.p[tsp["dytm"]["mv"]]

            # Add daytime charging location demand
            dytm_dist = getattr(self.rng, self.rd["dytm_dist"])
            self.dmnd.dytm.mv = np.array(
                [
                    pp * lds0[i] * dytm_dist(*self.rd["dytm_dist_params"]) / max(pp)
                    for i in self.ldsi.dytm.mv
                ]
            )

        # solar lv profiles [negative as these are demands]
        solar_rng = getattr(self.rng, self.rd["solar_dist"])
        if check_val("slr", "lv"):
            pp = self.p[tsp["slr"]["lv"]]
            if tsp["slr"]["lv"] == "solar0":
                slr0 = solar_rng(*self.rd["solar_dist_params"], size=self.ldsi.slr.nlv)
            self.dmnd.slr.lv = -np.outer(slr0, pp)

        # solar mv profiles
        if check_val("slr", "mv"):
            # It seems self.rng does not have an obvious way of returning
            # the mean programmatically.
            solar_mean = np.mean(solar_rng(*self.rd["solar_dist_params"], size=10000))

            # First infer the fraction of solar LV demands (avoids using
            # self.slr_pen, in case this option isn't used)
            pp = self.p[tsp["slr"]["mv"]]
            frac = self.ldsi.slr.nlv / self.mvlvi.nlv

            # Then, set the solar values.
            if tsp["slr"]["mv"] == "solar0":
                slr0 = np.array(
                    [frac * nhses[i] * solar_mean for i in self.ldsi.slr.mv]
                )
            self.dmnd.slr.mv = -np.outer(
                slr0,
                pp,
            )

        # DG mv profiles
        if type(tsp["dgs"]["mv"]) is list:
            # only solar0 implemented so far as a profile.
            assert tsp["dgs"]["mv"][0] == "solar0"
            self.dmnd.dgs.mv = -np.outer(tsp["dgs"]["mv"][1], self.p.solar0)

        # FCS mv profiles - based on DG profiles
        if check_val("fcs", "mv"):
            if tsp["fcs"]["mv"] == "usb_fcs":
                imax = np.argsort(np.max(self.p.usb_fcs, axis=1))
                self.dmnd.fcs.mv = self.p.usb_fcs[imax][-self.ldsi.fcs.nmv :]

            if type(tsp["fcs"]["mv"]) is list:
                # only solar0 implemented so far as a profile.
                assert tsp["fcs"]["mv"][0] == "uss24_urban_"
                self.dmnd.fcs.mv = np.outer(
                    tsp["fcs"]["mv"][1], self.p.uss24_urban_
                ) / max(self.p.uss24_urban_)

        # Heat pump LV profiles (at the moment these are only 1-d)
        if check_val("hps", "lv"):
            self.dmnd.hps.lv = np.outer(
                np.ones(self.ldsi.hps.nlv), self.p[tsp["hps"]["lv"]]
            )

        # hps mv profiles
        if check_val("hps", "mv"):
            frac = self.ldsi.hps.nlv / self.mvlvi.nlv
            pp = self.p[tsp["hps"]["mv"]]
            self.dmnd.hps.mv = np.outer(
                frac * nhses[self.ldsi.hps.mv],
                pp,
            )

    @staticmethod
    def load_solar0(
        nn=144,
    ):
        """Load the solar0 data as normalised data.

        Returns as 5*(288/nn) minute resolution data.
        """
        fn = os.path.join(
            data_dir,
            "solar-profile",
            "Actual_41.55_-74.25_2006_UPV_101MW_5_Min_xmpl.csv",
        )
        head, data = csvIn(fn)
        profile = np.array([float(r[head.index("Power(MW)")]) for r in data])
        return dds(profile / 101, 288 // nn)

    @staticmethod
    def load_ee_data(
        nn=144,
    ):
        """Load the Element Energy charging profiles.

        Values returned in kW at a 10*(144/nn) minute resolution.
        """
        fnev = os.path.join(data_dir, "ev-profile-data")
        rsev_, icev_ = [
            csvIn(os.path.join(fnev, f"{ss}.csv"), hh=False)
            for ss in ["rsev-week", "icev-week"]
        ]

        rsev, icev = [
            np.array(dd).astype("float")[:144, 1] / 1e3 for dd in [rsev_, icev_]
        ]
        return [dds(x, 144 // nn) for x in [rsev, icev]]

    @staticmethod
    def load_usb_fcs_data(
        nn=144,
    ):
        """Load the fast charger station data from the USB.

        As per email 10/12/21; these are in kW.
        """
        nday = 48
        fnfcs = os.path.join(
            data_dir,
            "power_series.csv",
        )
        data = np.array([r[1:] for r in csvIn(fnfcs)[1]]).astype(float)
        return dds(
            data,
            nday // nn,
        ).T

    @staticmethod
    def load_hp_profiles(
        nn=144,
    ):
        """Load nominal heat pump profiles."""
        fns = get_path_files(
            os.path.join(
                data_dir,
                "heat-pumps",
            ),
            ext="csv",
        )
        nday = 720
        data = {}
        for fn in fns:
            nm = os.path.basename(fn).split(".")[0]
            data_ = np.array([float(r[1]) for r in csvIn(fn, hh=False)])
            data[nm] = dds(np.r_[data_, data_][:720], nday // nn)

        return data

    @staticmethod
    def load_hp_love_profiles(
        nn=144,
    ):
        """Load the first day of the HP profiles from Love et al.

        Note that this also includes a step to remove nans, replacing them with
        the mean power of the month.
        """
        fns = get_path_files(
            os.path.join(
                data_dir,
                "heat-pumps",
                "IndividualProfiles",
            ),
            ext="csv",
        )
        day_sel = 15  # th of the month. Is a weekday in 2014.
        nday = 48
        data = {}
        for fn in fns:
            nm = os.path.basename(fn).split(".")[0]
            head, data_raw = csvIn(fn)
            data_ = np.array([r[1:] for r in data_raw], dtype=float).T / 1e3
            dsel = data_[:, nday * day_sel : (day_sel + 1) * nday]

            # replace nans in that day
            for i in range(len(dsel)):
                dsel[i, np.isnan(dsel[i])] = np.nanmean(data_[i])

            # Then assign to data.
            data[nm] = (
                dds(dsel.T, nday // nn).T
                if nday // nn >= 1
                else xtnd_ts(dsel, nn // nday)
            )

        return data

    @staticmethod
    def load_ev_urbanprofiles(
        nn=144,
    ):
        """Load the EV urbanprofiles datasets.

        Note these are only at hourly resolution; if the number of points to run
        is at a higher time resolution than this, then the response is assumed
        to just be on/off continuously (e.g., if 2 kW at 4pm, is 2 kW right from
        4pm to 5pm).
        """
        _, data_ = csvIn(os.path.join(data_dir, "EV_urbanprofiles.csv"))
        data = np.array(data_).astype(float)[:, 1:].T

        nday = 24
        if nday // nn < 1:
            return xtnd_ts(
                data,
                nn // nday,
            )
        else:
            return dds(data.T, nday // nn).T

    @staticmethod
    def load_ev_acn(
        nn=144,
        nwks=2,
    ):
        """Load nwks weekday EV profiles from the ACN [caltech] data.

        Is primarily data that can be used for 'daytime' charging.

        Source of the data:
        https://ev.caltech.edu/dataset
        """
        fn = os.path.join(
            data_dir,
            "ev-profile-data",
            "EV_ACN-21-07-27",
            "ACN_caltech--start-20210101-000000--end-20210630-000000.csv",
        )
        head, data = csvIn(fn)

        dts = [datetime.fromisoformat("2021-" + k.split("_2021-")[1]) for k in head[1:]]
        clk = np.arange(
            min(dts).date(),
            max(dts).date() + timedelta(1),
        ).astype(datetime)
        clk_i = np.array([np.argmax(clk == dd.date()) for dd in dts])
        clk_n = [sum(clk_i == i) for i in range(len(clk))]

        # Select five x nwks weekdays close to the maximum number per day
        imax = np.argmax(clk_n)
        isel = [i for i in range(imax, imax + (nwks * 9)) if clk[i].weekday() < 5][
            : nwks * 5
        ]
        i_dsel = [i for i, r in enumerate(clk_i) if r in isel]
        d_out = np.array([[r[1:][i] for i in i_dsel] for r in data], dtype=float).T

        nday = d_out.shape[1]
        if nday // nn < 1:
            return xtnd_ts(
                d_out,
                nn // nday,
            )
        else:
            return dds(d_out.T, nday // nn).T

        # # Some checksums
        # plt.plot(clk_n); tlps()
        # # Check the EVSE IDs [see Slack with Pam 27/7/21]
        # ids = [k.split('_2021-')[0][5:] for k in head[1:]]
        # unq_ids = set(ids)
        # n_se len(unq_ids)

    def load_ev_encc(
        self,
        nn=144,
        nwks=2,
    ):
        """Load Electric Nation profiles for nwks of weekdays.

        Loosely based on the function load_ev_acn.

        In this, there is a little error correction, where powers greater than
        the power of the charger are set to the maximum charger power.

        To ensure we are in Trail 1 [see page 17 of
        https://www.westernpower.co.uk/downloads-view-reciteme/64378 ] we just
        use 60 days in (early March 2018).
        """
        # Load the 2018 data.
        fns = [
            fn
            for fn in get_path_files(
                os.path.join(
                    data_dir,
                    "ev-profile-data",
                    "EV_WPD_Electric_Nation-21-07-27",
                )
            )
            if "-2018-" in fn
        ]
        [head0, data0], [head1, data1] = [csvIn(fn) for fn in fns]
        head = head0[1:] + head1[1:]

        # Get the data matrices, including a little error correction.
        d0, d1 = [np.array([d[1:] for d in dd], dtype=float) for dd in [data0, data1]]
        d0[d0 > 3.8] = 3.8
        d1[d1 > 7.5] = 7.5
        data = np.c_[d0, d1].T

        h2d = lambda h: datetime(*[int(v) for v in (h.split("_")[1].split("-"))])
        dts = np.array([h2d(h) for h in head])
        clk = np.arange(min(dts), max(dts) + timedelta(1), timedelta(1), dtype=datetime)

        # Find the number of charging times per day
        di = (dts - min(dts)) // timedelta(1)
        # unq,cnts = np.unique(di, return_counts=True,) # old version
        # imax = unq[np.argmax(cnts)] # old version

        # Pick early March to start, to avoid Trial 2
        imax = 60

        # Select nwks worth of data from those
        isel = [i for i in range(imax, imax + (nwks * 9)) if clk[i].weekday() < 5][
            : nwks * 5
        ]
        idsel = np.array([j for j, v in enumerate(di) if v in isel])

        # Shuffle the indexes so there is a mix of 3.6 & 7 kW chargers.
        self.rng.shuffle(idsel)
        d_out = data[idsel]

        nday = d_out.shape[1]
        if nday // nn < 1:
            return xtnd_ts(
                d_out,
                nn // nday,
            )
        else:
            return dds(d_out.T, nday // nn).T

    @staticmethod
    def load_csv_in(
        nn=144,
    ):
        """Load the csv_in data."""
        fn = os.path.join(
            data_dir,
            "csv_in.csv",
        )
        head, data_ = csvIn(fn)
        data = np.array(data_).astype(float)[:, 1:].T

        nday = 24
        if nday // nn < 1:
            return xtnd_ts(
                data,
                nn // nday,
            )
        else:
            return dds(data.T, nday // nn).T

    def print_lv_ckts(
        self,
        pp=True,
    ):
        """Print LV circuit info. Use pp flag to print to terminal.

        Returns
        ---
        ss - a string that can be printed print(ss)
        [head, data] - the heading and data that becomes ss.
        """
        tbl_info = odict(
            {
                "ldNo": "Load No.",
                "mlvNsplt": "No. Customers",
                "lvFrid": "LV Ckt. ID",
                "ctype": "No. Feeders",
            }
        )

        from tabulate import tabulate

        head = list(tbl_info.values())
        get_data = lambda k, i: self.ckts[k][i] if k != "mlvNsplt" else self.mlvNsplt[i]
        data = [[get_data(k, i) for k in tbl_info.keys()] for i in range(self.ckts.N)]

        ss = tabulate(data, headers=head)
        if pp:
            print(ss)

        return ss, [head, data]

    @staticmethod
    def load_uss24_urban(
        nn=144,
    ):
        """Load the smart meter data from the CLNR project."""
        fn = os.path.join(data_dir, "Book2_uss24_urban_over30k.csv")
        head, data_ = csvIn(fn)
        data = np.array(data_).astype(float)[:, 1:].T

        nday = 24
        if nday // nn < 1:
            return xtnd_ts(
                data,
                nn // nday,
            )
        else:
            return dds(data.T, nday // nn).T

    def record_solution(
        self,
        opts=None,
    ):
        """Record a solution for further visualization.

        Loosely based on self.getSln

        Inputs
        ---
        opts - None, not used rn

        Returns
        ---
        A sln Bunch (powers in kW/kVA; voltages/currents complex (A,V or pu)):
        - Spri, power through primary substation (tot system)
        - Ssec, power through MV/LV transformers

        - Ifmv, current through MV feeder (per phase), A
        - Iflv, current through LV feeder (list of lists, per phase), A
        - Sfmv, power through each MV feeder (per phase), kVA

        - Ltot, total system losses, kW

        - Vmv, complex voltage at all MV buses (indexed by self.mvIdx)
        - Vsb, pos sequence voltage at all LV secondary substations
        - VlvLds, complex voltage at all LV loads [list of vectors,
                                                indexed by self.ckts.ldNo]
        - VmvLds, complex voltage at all MV loads [indexed by self.ldsi.ic]

        - Tap, tap position(s)
        - Cnvg, convergence flag
        - Slds, power of all loads when solved solution
        - VsrcMeas, voltage source output voltage
        """
        sln = Bunch(
            mtDict(
                [
                    "Spri",
                    "Ssec",
                    "Ltot",
                    "Vmv",
                    "Vsb",
                    "Vld",
                    "Vic",
                    "Tap",
                    "Cnvg",
                    "Slds",
                    "VsrcMeas",
                ]
            )
        )

        tp2arL = lambda xx: [tp2ar(x) for x in xx]
        rm1 = lambda xx: [x[:-1] for x in xx]  # for removing ground voltages

        # Primary sub power
        sln.Spri = -tp2ar(d.DSSCircuit.TotalPower)[0]  # kW

        # Secondary sub powers
        ppsdry = tp2arL(d.getObjAttr(d.TRN, val=None, AEval="SeqPowers"))
        sln.Ssec = np.array([p[1] for p in ppsdry])[self.xfmri.sdry]

        # MV feeder currents + powers
        II2I = lambda II, idxs: np.array([ii[:3] for ii in vecSlc(II, idxs)])
        Ilns = tp2arL(d.getObjAttr(d.LNS, val=None, AEval="Currents"))
        sln.Ifmv = II2I(Ilns, self.pmryLnsi)
        Slns = tp2arL(d.getObjAttr(d.LNS, val=None, AEval="Powers"))
        sln.Sfmv = II2I(Slns, self.pmryLnsi)

        # LV feeder currents
        sln.Iflv = odict([[k, II2I(Ilns, v)] for k, v in self.sdryLnsi.items()])

        # Losses
        sln.Ltot = tp2ar(d.DSSCircuit.Losses)[0] / 1e3  # kW

        # Voltages at all MV buses
        sln.Vmv = tp2ar(d.DSSCircuit.YNodeVarray)[3:][self.mvIdx]

        # Voltages on the MV and LV Loads
        vlds = rm1(tp2arL(d.getObjAttr(d.LDS, val=None, AEval="Voltages")))

        sln.VmvLds = np.array(vecSlc(vlds, self.mvlvi.mv))
        sln.VlvLds = [np.array(vecSlc(vlds, v)) for v in self.mvlvi.lv_n.values()]

        # There are a few networks with 3-phase loads; pick phase A for ease
        obj2array = lambda obj: o2o(np.array([v[0] for v in obj]))
        _ = [
            sln.VlvLds.__setitem__(i, obj2array(xx))
            for i, xx in enumerate(sln.VlvLds)
            if xx.dtype == "O"
        ]

        # LV secondary substation voltages
        vsub = [v[4] for v in d.getObjAttr(d.TRN, val=None, AEval="SeqVoltages")]
        sln.Vsb = np.array(vsub)[self.xfmri.sdry]

        # Misc - for debugging
        sln.Tap = d.getTapPos()
        sln.Cnvg = d.SLN.Converged
        sln.Slds = rm1(tp2arL(d.getObjAttr(d.LDS, val=None, AEval="Powers")))
        sln.VsrcMeas = tp2ar(d.getObjAttr(d.Vsrcs, val=None, AEval="Voltages")[0])[
            :3
        ]  # in V

        return sln

    def get_lds_kva(
        self,
        tsp_n,
        dtypes=None,
    ):
        """Determine the d.LDS demand (kw) for running with OpenDSS.

        Uses the matrices determined in self.set_load_profiles.

        Inputs
        ---
        tsp_n; the ts_profiles data 'n'
        dtypes; if None the uses all of self.ldsi.kw; otherwise, pass in a
                list of which of these would like to be included in
                matrix returned.

        Returns
        ---
        lds; a nlds x tsp_n matrix of demands, with the row index
            corresponding to the ith load (in d.LDS, when iterated over [i.e.,
            when looping through, not necessarily the load that will be set
            due to disabled loads.])
        """
        dtypes = self.ldsi.kw if dtypes is None else dtypes

        # Get no. lds - d.LDS.Count does NOT work here due to disabled loads
        nlds = len(
            d.getObjAttr(
                d.LDS,
                "kva",
            )
        )
        lds = np.zeros(
            (
                nlds,
                tsp_n,
            )
        )
        for dtype in dtypes:
            for lvl in self.str_lvmv:
                lds[self.ldsi[dtype][lvl]] += self.dmnd[dtype][lvl]

        return lds

    def run_dss_lds(
        self,
        lds,
    ):
        """Run a time series analysis using nlds x ntime matrix lds.

        Inputs
        ---
        lds - the loads matrix to set d.LDS with using d.setObjAttr

        Returns
        ---
        slns - a list of solutions (see help(self.record_solution) )
        sln0 - the solution prior to running the analysis

        """
        # First record the initial solution
        sln0 = self.record_solution()

        # Then run through each row of lds and solve.
        slns = []
        with Bar("Running OpenDSS", suffix="%(percent).1f%% - %(eta)ds") as bar:
            for i, ldval in enumerate(lds.T):
                d.setObjAttr(d.LDS, "kva", ldval)
                d.SLN.Solve()
                slns.append(self.record_solution())
                bar.next(101 / len(lds.T))

        return slns, sln0


class modify_network:
    """Class to take the params of run_dict and create a working dss model.

    The approach is to manually modify a DSS network and save in a temporary folder in ./networks.

    Method:
    - Copy the main network files to the dnout directory (not the LV networks);
    - Modify the relative paths in those files so that they point to the LV
        networks;
    - Re-enable the lump loads on the MV system that do not have LV networks.

    """

    def __init__(self, run_dict, mod_dir=os.getcwd(), dnout="_network_mod"):
        logging.info("Entering modify_network __init__")

        # output directory (in ./networks)
        n_id = run_dict["network_data"]["n_id"]

        self.mod_dir = mod_dir  # Location for temp files - unzipped, modified etc.
        self.dnout = dnout  # Stub name for modified files
        self.fldr_ntwk = os.path.dirname(snk.dNet.fdrsLoc[n_id])
        # self.source = os.path.join(self.mod_dir, "networks", self.fldr_ntwk)
        self.destination = os.path.join(mod_dir, dnout)  # Full path of modified files

        # First, cleanup then initialise the dnout directory
        # self.initialise_directory()

        # Then, copy the files from the MV-LV network models over;
        # self.copy_network_files()

        # Finally modify the dss files according the options in run_dict
        self.modify_dss_files(
            run_dict["network_data"],
        )

    # # Commenting out initialise_directory() and copy_network_files() as they aren't used with our current method
    # # of editing the network files, but we may want to bring them back if we move the networks out of blob storage
    # # and back into this repo.

    # def initialise_directory(
    #     self,
    # ):
    #     """Create/cleanup the directory self.dnout in ./networks ."""
    #     dn = os.path.join(fn_root, "networks", self.dnout)
    #     if os.path.exists(dn):
    #         shutil.rmtree(dn, ignore_errors=True)

    #     _ = os.mkdir(dn) if not os.path.exists(dn) else None
    #     print(os.path.exists(dn))

    # def copy_network_files(
    #     self,
    #     n_id,
    # ):
    #     """Copy the master file in the network directory.

    #     Inputs
    #     ---
    #     n_id - the ID of the MVLV network to use.

    #     """
    #     # Get the source/destination directories
    #     self.fldr_ntwk = os.path.dirname(snk.dNet.fdrsLoc[n_id])
    #     dn_src = os.path.join(fn_root, "networks", self.fldr_ntwk)
    #     dn_dst = os.path.join(
    #         fn_root,
    #         "networks",
    #         self.dnout,
    #     )

    #     # Copy over the files.
    #     fn_names = get_path_files(
    #         dn_src,
    #         mode="names",
    #     )
    #     for fn in fn_names:
    #         shutil.copy(os.path.join(dn_src, fn), os.path.join(dn_dst, fn))

    # self.dn_dst = dn_dst

    def modify_dss_files(
        self,
        nd,
    ):
        """Modify the files in the new directory as specified by nd.

        Inputs
        ---
        nd - 'network_dict' from runDict.

        """

        logging.info("Entering modify_dss_files")

        # copy master, loads, and lvNetworksRedirect files;
        fn_copy = {
            "lds": "lds_edit",
            "rgc": "regcontrols",
            "xfmr": "transformers",
            "mstr": "master_mvlv",
            "lv": "redirect_lv_ntwx",
        }
        # for fn in fn_copy.values():
        #     fn_src = os.path.join(self.destination, fn)
        #     fn_dst = fn_src + self.dnout
        #     shutil.copy(fn_src+'.dss', fn_dst+'.dss')

        # # Get resulting file names
        # fn_lds,fn_mstr,fn_lv,fn_rgc,fn_xfmr = [os.path.join(
        #         self.destination,fn_copy[vv]+'.dss')
        #                                 for vv in ['lds','mstr','lv','rgc','xfmr']]

        fn_lds = os.path.join(self.mod_dir, self.dnout, fn_copy["lds"] + ".dss")
        fn_lv = os.path.join(self.mod_dir, self.dnout, fn_copy["lv"] + ".dss")
        fn_rgc = os.path.join(self.mod_dir, self.dnout, fn_copy["rgc"] + ".dss")
        fn_xfmr = os.path.join(self.mod_dir, self.dnout, fn_copy["xfmr"] + ".dss")

        base_folder = os.path.join(self.mod_dir, self.dnout)

        # # Update the master file
        # with open(fn_mstr, "r") as file:
        #     mstr_txt = file.read()

        # mstr_txt = mstr_txt.replace(
        #     "redirect redirect_lv_ntwx.dss",
        #     f"!redirect redirect_lv_ntwx.dss\nredirect redirect_lv_ntwx{self.dnout}.dss",
        # )

        # mstr_txt = mstr_txt.replace(
        #     "redirect lds_edit.dss",
        #     f"!redirect lds_edit.dss\nredirect lds_edit{self.dnout}.dss",
        # )

        # mstr_txt = mstr_txt.replace(
        #     "Redirect transformers.dss",
        #     f"!Redirect transformers.dss\nRedirect transformers{self.dnout}.dss",
        # )

        # mstr_txt = mstr_txt.replace(
        #     "Redirect regcontrols.dss",
        #     f"!Redirect regcontrols.dss\nRedirect regcontrols{self.dnout}.dss",
        # )

        # with open(fn_mstr, "w") as file:
        #     file.write(mstr_txt)

        # Functions for getting the LV network ID from strings
        getLvi = lambda s: s.split("_")[-1][:-4]
        getLdi = lambda s: s.split(" ")[1].split("_")[-1]

        # Change the paths in the lvNetworks file and update those that are 'in'
        with open(fn_lv, "r") as file:
            lv_txt = file.read()

        # nlvTot = lv_txt.count('redirect')
        lv_relpath = os.path.relpath(
            os.path.join(self.mod_dir, self.dnout, "lvNetworks"), self.destination
        )
        nlvTot = lv_txt.count("\n")
        lv_txt = lv_txt.replace("lvNetworks", lv_relpath)
        lv_lst = lv_txt.split("\n")

        # Get the LV networks which will be modelled in full:
        if nd["lv_sel"] == "n_lv":
            II = np.arange(nd["n_lv"], nlvTot)
        elif nd["lv_sel"] == "lv_ilist":
            II = [i for i in range(nlvTot) if i not in nd["lv_ilist"]]
        elif nd["lv_sel"] == "lv_list" or nd["lv_sel"] in [
            "near_sub",
            "near_edge",
            "mixed",
        ]:
            lv_list = nd["lv_list"] if nd["lv_sel"] == "lv_list" else nd[nd["lv_sel"]]
            lv_idxs = [getLvi(rr) for rr in lv_lst]
            lvl_idxs = [lv_idxs.index(ii) for ii in lv_list]
            II = [i for i in range(nlvTot) if i not in lvl_idxs]

        lv_lst = [
            s.replace("redirect", "!redirect") if i in II else s
            for i, s in enumerate(lv_lst)
        ]
        IIlds = [getLvi(s) for s in lv_lst if "!redirect" in s]

        with open(fn_lv, "w") as file:
            file.write("\n".join(lv_lst))

        # Enable/disable the loads on disabled networks
        with open(fn_lds, "r") as file:
            lds_txt = file.read()

        lds_lst = [
            s.replace("False", "True") if getLdi(s) in IIlds else s
            for s in lds_txt.split("\n")[:-1]
        ]

        with open(fn_lds, "w") as file:
            file.write("\n".join(lds_lst))

        # Update the substation definition and regcontrols
        with open(fn_rgc, "r") as file:
            rgc_txt = file.read()

        rgc_lst = rgc_txt.split("\n")[:-1]
        rgc_lst = [
            self.dss_value_update(s, "vreg", val=str(nd["oltc_setpoint"]))
            for s in rgc_lst
        ]
        rgc_lst = [
            self.dss_value_update(s, "band", val=str(nd["oltc_bandwidth"]))
            for s in rgc_lst
        ]

        with open(fn_rgc, "w") as file:
            file.write("\n".join(rgc_lst))

        # Update the transformer ratings
        with open(fn_xfmr, "r") as file:
            xfmr_txt = file.read()

        xfmr_lst = [
            self.dss_value_update(s, "kVA", val_mult=nd["xfmr_scale"])
            for s in xfmr_txt.split("\n")[:-1]
        ]

        with open(fn_xfmr, "w") as file:
            file.write("\n".join(xfmr_lst))

        logging.info("Leaving modify_dss_files")

    @staticmethod
    def dss_value_update(
        ss,
        kk,
        val=None,
        val_mult=None,
    ):
        """Update the string ss with key kk to new value val.

        Inputs
        ---
        ss - string to be updated (typically a line of OpenDSS code)
        kk - the key to find
        val - if not None, the string to change kk=val to
        val_mult - if not None, the float to multiply kk=val by.
        """
        assert val is None or val_mult is None
        kvs = [v.split("=") for v in ss.split(" ")]

        # Make the changes
        for kv in kvs:
            if kv[0] == kk:
                if not (val is None or val == "None"):
                    kv[1] = val
                elif not val_mult is None:
                    kv[1] = str(val_mult * float(kv[1]))

        ss = " ".join("=".join(kv) for kv in kvs)
        return ss


def unzip_networks(dest_dir, n_id):
    """Unzip the networks in ./networks/ if any need doing."""

    logging.info("Entering unzip_networks")

    # Unzip the network zip file if not already
    # Connect to blob service, yes hard-coded is bad
    blob_service_client = BlobServiceClient.from_connection_string(
        get_settings().networks_data_container_readonly_connection_string
    )
    container = get_settings().networks_data_container_readonly

    zip_names = {1060: "HV_UG_full.zip", 1061: "HV_UG-OHa_full.zip"}

    # Connect to 'apidata' container which has the networks and no more
    # It should be renamed to 'networks_store' or something
    container_client = blob_service_client.get_container_client(container=container)
    logging.info("Unzipping files to: %s", dest_dir)

    ntwk_name = zip_names[n_id]

    # Connect to the blob itself
    blob_client = blob_service_client.get_blob_client(
        container=container, blob=ntwk_name
    )
    # Download the network zip as data, not written to disk
    ntwk_data = blob_client.download_blob().readall()

    unzipped_folder = os.path.join(dest_dir, ntwk_name.split(".")[0])

    if not os.path.exists(unzipped_folder):
        with zipfile.ZipFile(BytesIO(ntwk_data), "r") as zip_ref:
            zip_ref.extractall(dest_dir)

            # Check directory separator style and edit from \ -> / if necessary
            # dss_utils.changeDirectorySeparatorStyle(os.path.join(dest_dir, ntwk.name.split(".")[0]), verbose=True)

    del ntwk_data
    gc.collect()

    # Basic checking numbers - basic, might need updating in future for,
    # e.g., frIds above 100
    installed_names = get_path_dirs(
        dest_dir,
        mode="names",
    )
    dn2frids = snk.dNet.dn2frids
    dn2frids.update(
        {
            "manchester_models": "101-125 incl.",
        }
    )
    frids = [dn2frids.get(nm, "na") for nm in installed_names]

    print("\n---------\nFilename, (frId, if applicable)\n---------")
    _ = [print(f"{nm} ({id})") for nm, id in zip(installed_names, frids)]
    print("")

    logging.info("Leaving unzip_networks")
