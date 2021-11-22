import os, sys, dss, shutil
from . import dss_utils

import os, sys, pickle, zipfile
from pprint import pprint
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from timeit import default_timer as timer
from collections import OrderedDict as odict
from copy import deepcopy
import logging

from bunch import Bunch

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
)
from . import funcsDss_turing
from .funcsMath_turing import vecSlc, tp2ar
from . import slesNtwk_turing as snk

from . import azureOptsXmpls as aox

from io import BytesIO
from azure.storage.blob import BlobServiceClient
import gc
from .config import get_settings

fn_root = sys.path[0] if __name__ == "__main__" else os.path.dirname(__file__)

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


class turingNet(snk.dNet):
    """A MLV class for development of code alongside V2G with Turing."""

    def __init__(
        self,
        frId=40,
        frId0=None,
        rundict={},
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
        snk.mlvNet.getLvCktInfo(self, fn0=os.path.join(fn0, "lvNetworks"))
        # snk.mlvNet.setAuxCktAttributes(self,) # Not too sure if needed?

        self.set_base_tags()
        self.set_ldsi(
            dgd=rundict["dmnd_gen_data"],
        )
        self.set_load_profiles(
            tsp=rundict["simulation_data"]["ts_profiles"],
        )

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

        # Cache some getObjectAttr values
        objattrcache = {}
        objattrcache["LDS"] = d.getObjAttrDict(d.LDS, ["kV", "kW", "Name"])
        objattrcache["TRN"] = d.getObjAttrDict(d.TRN, ["Name"], ["BusNames"])
        objattrcache["LNS"] = d.getObjAttrDict(d.LNS, ["Bus1"])

        # First splite loads into LV (<1 kV) and MV (otherwise)
        kVkW = listT([objattrcache["LDS"]["kV"], objattrcache["LDS"]["kW"]])
        self.mvlvi = Bunch(
            {
                "mv": [i for i, (v, p) in enumerate(kVkW) if v > 1.0 and p != 0],
                "lv": [i for i, (v, p) in enumerate(kVkW) if v <= 1.0],
            }
        )
        self.mvlvi.nmv = len(self.mvlvi.mv)
        self.mvlvi.nlv = len(self.mvlvi.lv)

        # Create an OrderedDict for the residential loads split by ckt
        ldn = objattrcache["LDS"]["Name"]
        self.mvlvi.lv_n = mtOdict(self.ckts.ldNo)
        for i, n in enumerate(ldn):
            if n.count("_") == 3:
                self.mvlvi.lv_n[n.split("_")[1]].append(i)

        # Similarly find the transformer voltages
        xnms = objattrcache["TRN"]["Name"]
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
            i for i, b in enumerate(objattrcache["LNS"]["Bus1"]) if b == pmryBus
        ]

        # Then get the buses of all secondary LV substations
        trnBuses = objattrcache["TRN"]["BusNames"]
        sdryBus = [trnBuses[i][1] for i in self.xfmri.sdry]

        # ...and subsequently the indexes of lines that connect to them
        self.sdryLnsi = mtOdict(self.ckts.ldNo)
        for i, lnb in enumerate(objattrcache["LNS"]["Bus1"]):
            if lnb in sdryBus:
                self.sdryLnsi[lnb.split("_")[1]].append(i)

    def set_ldsi(
        self,
        dgd={},
    ):
        """Set ldsi, the indexes for each demand type using dmnd_gen_data (dgd).

        Flags which are set (via a list of indexes in d.LDS):
        - residential (LV, MV lumped): rslv, rsmv;
        - I&C (LV, MV lumped): iclv, icmv;
        - daytime / overnight charging: dytm, ovnt;
        - solar PV: slr;
        - heat pump: hps;

        Options in dgd (all can be set to None):
        - rs.lv + ic.lv:
            - 'lv' (all lv)
        - rs.mv + ic.mv
            - 'mv' (all mv)
            - 'odds'/'evens' (either all 'odd' or all 'even' indexes)
        - ovnt/dytm:
            - 'rs' (all residential),
            - 'ic' (all I&C)
        - slr:
            - None [to implement]
        - hps:
            - None [to implement]

        TO DO - it would be worthwhile adding some assertions here so that only
        things that 'make sense' are options that work and otherwise things
        a problem is flagged.
        """
        # Initialise and set all values with None to be empty.
        self.ldsi = Bunch({})
        _ = [setattr(self.ldsi, k, Bunch({"lv": [], "mv": []})) for k in dgd.keys()]
        # for k,v in dgd.items() if v is None]
        lvmv = [
            "lv",
            "mv",
        ]
        self.ldsi.kw = list(dgd.keys())

        for vv in lvmv:
            # First set residential loads:
            if dgd["rs"][vv] == vv:
                self.ldsi.rs[vv] = deepcopy(self.mvlvi[vv])
            elif dgd["rs"][vv] in ["odds", "evens"]:
                offset = 0 if dgd["rs"][vv] == "evens" else 1
                self.ldsi.rs[vv] = deepcopy(self.mvlvi[vv])[offset::2]

            # Then set I&C loads:
            if dgd["ic"][vv] == vv:
                self.ldsi.ic[vv] = deepcopy(self.mvlvi[vv])
            elif dgd["ic"][vv] in ["odds", "evens"]:
                offset = 0 if dgd["ic"][vv] == "evens" else 1
                self.ldsi.ic[vv] = deepcopy(self.mvlvi[vv])[offset::2]

            # Fix the overnight / daytime charging profiles
            if dgd["dytm"][vv] == "ic":
                self.ldsi.dytm[vv] = deepcopy(self.ldsi.ic[vv])
            elif dgd["dytm"][vv] == "rs":
                self.ldsi.dytm[vv] = deepcopy(self.ldsi.rs[vv])

            if dgd["ovnt"] == "ic":
                self.ldsi.ovnt[vv] = deepcopy(self.ldsi.ic[vv])
            elif dgd["ovnt"][vv] == "rs":
                self.ldsi.ovnt[vv] = deepcopy(self.ldsi.rs[vv])

            # Solar
            pass

            # Heat pumps
            pass

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
        txtFs = pnkw.get(
            "txtFs",
            10,
        )

        lvn_clr = "r"
        lvn_dict = {} if not pType == "B" else {"facecolor": lvn_clr, "alpha": 0.3}

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
            icMvBuses = self.ldsi2buses(self.ldsi.ic.mv)
            rsMvBuses = self.ldsi2buses(self.ldsi.rs.mv)
            lvNtwkBuses = self.ckts.ldNo
            aln = ["top", "bottom"] * (d.LDS.Count // 2)
            # fig,ax = plt.subplots(figsize=(13,7),)
            fig, ax = plt.subplots(
                figsize=(9, 5),
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
                ax_scat = self.plotLvNtwx(
                    ax,
                    lvn_clr,
                )

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
            lbls = [
                "Res. demand,\nlumped",
                "I&C demand,\nlumped",
                "Res. demand,\nLV Modelled",
            ]
            plt.legend(
                [plt_lv, plt_mv, (plt_lv, xLg)],
                lbls,
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

            # Cache some getObjectAttr values
            objattrcache = {}
            objattrcache["LDS"] = d.getObjAttrDict(d.LDS, ["kW", "kvar"])

            mvPwrs_S = np.array(objattrcache["LDS"]["kW"]) + 1j * np.array(
                objattrcache["LDS"]["kvar"]
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
                self.plotLvNtwx(
                    ax,
                    lvn_clr,
                )

            # Update so that only the voltage locations are plotted
            pType = "v"

        self.plotNetwork(pType, **pnkw)

    def plotLvNtwx(self, ax, lvn_clr):
        """Plot the LV networks that are modelled explicitly."""
        for bus in self.ckts.ldNo:
            ax_scat = ax.scatter(
                *self.busCoords[bus],
                s=self.fPrm_["pms"],
                facecolors="None",
                marker=".",
                zorder=30,
                edgecolors=lvn_clr,
                linewidth=1.2,
            )
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
            ax.text(
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
        dn0 = os.path.join(fn_root, "data", "coords")
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

    def set_load_profiles(
        self,
        tsp,
    ):
        """Load the underlying demand curves for all specified loads.

        Types set are specifed in self.ldsi.kw.

        Each is split into either a 'lumped' (mv) or 'individual' (lv) load.

        Each profile is indexed by an index i, with j the hourly demand. For
        now we are looking at 10-minute timesteps. The units of ALL of the
        values are in kW. Create the loads matrix using self.get_lds_kva.

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
        Otherwise:
        - rs.lv:
            - 'crest' - a set of generic profiles created by the CREST tool
            from Loughborough univeristy.
        - rs.mv:
            - 'crest_' - take the mean of the LV profiles from rs.lv 'crest'.
        - ic.lv:
            - pass
        - ic.mv:
            - 'ic00_prof', a I&C profile loosely based on I&C profiles from the
            T. Short's 'Electric Power Distribution Handbook', 2014, Ch 1.
        - ovnt.lv:
            - 'ee', estimated profile from Element Energy report (data in
            directory ./data/ev-profile-data).
        - ovnt.mv:
            - 'ee', same as above
        - dytm.lv
            - pass
        - dytm.mv
            - pass
        - slr.lv
            - pass
        - slr.mv
            - pass
        - hps.lv
            - pass
        - hps.mv
            - pass
        """
        # Initialise all as zeros of an appropriate dimension
        self.dmnd = Bunch(
            {
                k: Bunch(
                    {
                        vv: np.zeros((self.ldsi[k]["n" + vv], tsp["n"]))
                        for vv in [
                            "lv",
                            "mv",
                        ]
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

        # First load the CREST demand (in case it is used)
        fndemand = os.path.join(fn_root, "data", "ciredData.pkl")
        with open(fndemand, "rb") as file:
            crestData = pickle.load(file)

        ddsn = len(crestData[1]["peNet"]) // tsp["n"]  # downsample no
        r0 = dds(crestData[1]["peNet"], ddsn) / 1e3  # W to kW

        # Similarly load the Element Energy data
        rsev, icev = self.load_ee_data()
        assert len(rsev) == tsp["n"]
        assert len(icev) == tsp["n"]

        # Residential demand rs
        if tsp["rs"]["lv"] == "crest" and self.ldsi.rs.nlv > 0:
            # Loop through the crest data and assign profiles
            self.dmnd.rs.lv = np.array(
                [r0[:, i % r0.shape[1]] for i in range(self.ldsi.rs.nlv)]
            )

        if tsp["rs"]["mv"] == "crest_" and self.ldsi.rs.nmv > 0:
            # Assign the mean profile to all loads
            r00 = np.mean(r0, axis=1) / max(np.mean(r0, axis=1))
            self.dmnd.rs.mv = np.array([lds0[i] * r00 for i in self.ldsi.rs.mv])

        # I&C demand ic
        pass  # ic lv demand

        if tsp["ic"]["mv"] == "ic00_prof" and self.ldsi.ic.nmv > 0:
            # Assign the I&C profile
            ic00 = ic00_prof / max(ic00_prof)
            self.dmnd.ic.mv = np.array([lds0[i] * ic00 for i in self.ldsi.ic.mv])

        # Set the overnight charging profiles
        if tsp["ovnt"]["lv"] == "ee":
            self.dmnd.ovnt.lv = np.array([rsev for i in range(self.ldsi.ovnt.nlv)])

        pass  # overnight mv charging

        # Set the daytime charging profiles
        pass  # daytime lv charging

        if tsp["dytm"]["mv"] == "ee":
            # Add 10% of demand at I&C locations, assumed to be
            self.dmnd.dytm.mv = np.array(
                [icev * lds0[i] * 0.10 / max(icev) for i in self.ldsi.dytm.mv]
            )

        # Set the solar profiles
        pass  # solar lv profiles
        pass  # solar mv profiles

        # Set the heat pump profiles
        pass  # hps lv profiles
        pass  # hps mv profiles

    @staticmethod
    def load_ee_data():
        """Load the Element Energy charging profiles.

        Values returned in kW at a 10 mintue resolution.
        """
        fnev = os.path.join(fn_root, "data", "ev-profile-data")
        rsev_, icev_ = [
            csvIn(os.path.join(fnev, f"{ss}.csv"), hh=False)
            for ss in ["rsev-week", "icev-week"]
        ]
        rsev, icev = [
            np.array(dd).astype("float")[:144, 1] / 1e3 for dd in [rsev_, icev_]
        ]
        return rsev, icev

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

        - Ifmv, power through MV feeder (per phase)
        - Iflv, power through LV feeder (list of lists, per phase)

        - Ltot, total system losses

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

        # Cache some getObjectAttr values
        objattrcache = {}
        objattrcache["TRN"] = d.getObjAttrDict(d.TRN, [], ["SeqPowers", "SeqVoltages"])
        objattrcache["LDS"] = d.getObjAttrDict(d.LDS, [], ["Voltages", "Powers"])

        # Secondary sub powers
        ppsdry = tp2arL(objattrcache["TRN"]["SeqPowers"])
        sln.Ssec = np.array([p[1] for p in ppsdry])[self.xfmri.sdry]

        # MV feeder currents
        II2I = lambda II, idxs: np.array([ii[:3] for ii in vecSlc(II, idxs)])
        Ilns = tp2arL(d.getObjAttr(d.LNS, val=None, AEval="Currents"))
        sln.Ifmv = II2I(Ilns, self.pmryLnsi)

        # LV feeder currents
        sln.Iflv = odict([[k, II2I(Ilns, v)] for k, v in self.sdryLnsi.items()])

        # Losses
        sln.Ltot = tp2ar(d.DSSCircuit.Losses)[0] / 1e3  # kW

        # Voltages at all MV buses
        sln.Vmv = tp2ar(d.DSSCircuit.YNodeVarray)[3:][self.mvIdx]

        # Voltages on the MV and LV Loads
        vlds = rm1(tp2arL(objattrcache["LDS"]["Voltages"]))

        sln.VmvLds = np.array(vecSlc(vlds, self.mvlvi.mv))
        sln.VlvLds = [np.array(vecSlc(vlds, v)) for v in self.mvlvi.lv_n.values()]

        # LV secondary substation voltages
        vsub = [v[4] for v in objattrcache["TRN"]["SeqVoltages"]]
        sln.Vsb = np.array(vsub)[self.xfmri.sdry]

        # Misc - for debugging
        # sln.Tap = d.getTapPos()
        # sln.Cnvg = d.SLN.Converged
        # sln.Slds = rm1(tp2arL(objattrcache['LDS']['Powers']))
        # sln.VsrcMeas = tp2ar(
        #         d.getObjAttr(d.Vsrcs,val=None,AEval='Voltages')[0])[:3] # in V

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
            for lvl in [
                "lv",
                "mv",
            ]:
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
        for i, ldval in enumerate(lds.T):
            d.setObjAttr(d.LDS, "kva", ldval)
            d.SLN.Solve()
            slns.append(self.record_solution())

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

        logging.info("Leaving modify_network __init__")

    def initialise_directory(
        self,
    ):
        """Create/cleanup the directory self.dnout in ./networks ."""

        logging.info("Entering initialise_directory")

        if os.path.exists(self.destination):
            shutil.rmtree(self.destination, ignore_errors=True)

        _ = os.mkdir(self.destination) if not os.path.exists(self.destination) else None
        logging.info("Edited .dss files will be saved to %s", self.destination)

        logging.info("Leaving initialise_directory")

    def copy_network_files(self):
        """Copy the network directory."""

        logging.info("Entering copy_network_files")

        # Get the source/destination directories
        shutil.copytree(self.source, self.destination, dirs_exist_ok=True)

        logging.info("Leaving copy_network_files")

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
            "mstr": "master_mvlv",
            "lv": "redirect_lv_ntwx",
        }
        # for fn in fn_copy.values():
        #     fn_src = os.path.join(self.destination, fn)
        #     fn_dst = fn_src + self.dnout
        #     shutil.copy(fn_src+'.dss', fn_dst+'.dss')

        # # Get resulting file names
        # fn_lds,fn_mstr,fn_lv = [os.path.join(
        #         self.destination,fn_copy[vv]+'.dss')
        #                                 for vv in ['lds','mstr','lv']]

        # fn_lds,fn_mstr,fn_lv = [os.path.join(
        #         self.destination,fn_copy[vv]+'.dss')
        #                                 for vv in ['lds','mstr','lv']]

        fn_lds = os.path.join(self.mod_dir, self.dnout, fn_copy["lds"] + ".dss")
        fn_lv = os.path.join(self.mod_dir, self.dnout, fn_copy["lv"] + ".dss")
        logging.info("******{}".format(fn_lds))

        base_folder = os.path.join(self.mod_dir, self.dnout)
        logging.info(os.listdir(base_folder))

        # # Update the master file
        # with open(fn_mstr,'r') as file:
        #     mstr_txt = file.read()

        # mstr_txt = mstr_txt.replace(
        #     'redirect redirect_lv_ntwx.dss',
        #     f'!redirect redirect_lv_ntwx.dss\nredirect redirect_lv_ntwx{self.dnout}.dss',)

        # mstr_txt = mstr_txt.replace(
        #     'redirect lds_edit.dss',
        #     f'!redirect lds_edit.dss\nredirect lds_edit{self.dnout}.dss',)

        # with open(fn_mstr,'w') as file:
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
        elif nd["lv_sel"] == "lv_list":
            lv_idxs = [getLvi(rr) for rr in lv_lst]
            lvl_idxs = [lv_idxs.index(ii) for ii in nd["lv_list"]]
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

        logging.info("Leaving modify_dss_files")


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
