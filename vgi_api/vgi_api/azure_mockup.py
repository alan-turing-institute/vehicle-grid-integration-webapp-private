"""azure-mockup.py
A script for mocking up how the python functions can work with 
Azure. azureOptsXmpls.py contains some example dictionaries with the 
options that could be changed.

General approach:
- First, take the 'network_data' parameters which are used to modify the mvlv
  network models and use them to create a new network in _network_mod (given
  network ID 1000). This uses the ft.modify_network class.
- Then, take the 'simulation_data' parameters and run a simulation. This uses
  the turingNet class.

The full options and definitions of the run_dict are given in azureOptsXmpls.
As a quick hack, ppd(aox) lists the options available.
"""
import os
import dss
import io
import sys
from . import funcsTuring as ft
from . import funcsDss_turing
from . import azureOptsXmpls as aox
from .funcsPython_turing import gDir, fillplot, set_day_label
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import logging

fn_root = sys.path[0] if __name__ == "__main__" else os.path.dirname(__file__)


def run_dss_simulation(rd=aox.run_dict0, sf=0):

    logging.info("Entering run_dss_simulation")

    # Set up a temporary directory to store network files
    with tempfile.TemporaryDirectory() as temp_dir:
        ft.unzip_networks(
            dest_dir=os.path.join(temp_dir, "_network_mod"),
            n_id=rd["network_data"]["n_id"],
        )
        d = funcsDss_turing.dssIfc(dss.DSS)
        # Place modified files into _network_mod to match hardcoded value in slesNtwk_turing.py
        ntwk = ft.modify_network(rd, mod_dir=temp_dir, dnout="_network_mod")

        # Simulation modifications

        frid0 = rd["network_data"]["n_id"]
        simulation = ft.turingNet(
            frId=1000,
            frId0=frid0,
            rundict=rd,
            mod_dir=temp_dir
        )
        tsp_n = rd["simulation_data"]["ts_profiles"]["n"]
        tt = np.arange(0, 24, 24 / tsp_n)  # get the clock

        # Get the solutions
        lds = simulation.get_lds_kva(tsp_n)
        slns, _ = simulation.run_dss_lds(lds)


        # From here: plotting options
        if rd["plot_options"]["mv_highlevel"]:
            # Plot the network we have created
            simulation.fPrm.update(
                {
                    "saveFig": sf,
                    "sd": gDir,
                    "showFig": True,
                    "pdf": False,
                    "figname": "pltNetworks_mvonly_new",
                }
            )
            simulation.plotXvNetwork(
                pType="B",
                pnkw={"txtOpts": "all", "dgFlag": True},
            )

        if rd["plot_options"]["mv_highlevel_clean"]:
            # Plot the network we have created
            simulation.fPrm.update(
                {
                    "saveFig": sf,
                    "sd": gDir,
                    "showFig": True,
                    "pdf": False,
                    "figname": "pltNetworks_mvonly_new_clean",
                }
            )
            simulation.plotXvNetwork(
                pType="B",
                pnkw={"txtOpts": None, "dgFlag": False, "figsize": (7, 3.6)},
            )

        if rd["plot_options"]["mv_powers"]:
            # Power plot (no LV circles)
            simulation.fPrm.update(
                {
                    "saveFig": sf,
                    "sd": os.path.join(fn_root, "data", "mvlv_topos"),
                    "showFig": True,
                    "pdf": False,
                    "figname": f"{simulation.fdrs[frid0]}",
                }
            )
            txtFss = {
                1060: "10",
                1061: "6",
            }
            pnkw = {
                "txtOpts": "all",
                "lvnFlag": False,
                "txtFs": txtFss[frid0],
            }
            simulation.plotXvNetwork(
                pType="p",
                pnkw=pnkw,
            )

        if rd["plot_options"]["lv_voltages"][0]:
            assert len(rd["plot_options"]["lv_voltages"][1]) < 3  # for now only compare two profiles

            fig, ax = plt.subplots()
            lv_idxs = [simulation.ckts.ldNo.index(nn) for nn in rd["plot_options"]["lv_voltages"][1]]

            clrs = [
                "C0",
                "C3",
            ]
            for idx, clr in zip(lv_idxs, clrs):
                Vsec = np.array([s["VlvLds"][idx] for s in slns])[:, :, 0]
                fillplot(
                    np.abs(Vsec) / 230, tt, ax=ax, lineClrs=[clr], fillKwargs={"color": clr}
                )

            _ = [
                plt.plot(np.nan, np.nan, color=clr, label=lbl)
                for lbl, clr in zip(simulation.ckts.ldNo, clrs)
            ]
            plt.legend(
                title="LV ckt ID",
                fontsize="small",
            )
            set_day_label()
            plt.ylabel("Voltage, pu")
            xlm = plt.xlim()
            plt.hlines(
                [0.94, 1.10],
                *xlm,
                linestyles="dashed",
                color="r",
            )
            plt.xlim(xlm)
            if sf:
                sff("lv_voltages")

            tlps()

        if rd["plot_options"]["mv_voltage_ts"]:
            # Voltage plot against time
            smv2pu = lambda s: np.abs(s.Vmv) / simulation.vKvbase[simulation.mvIdx]
            vb = np.array([smv2pu(s) for s in slns])
            fillplot(vb, np.linspace(0, 24, tsp_n))
            set_day_label()
            xlm = plt.xlim()
            plt.hlines([0.94, 1.06], *xlm, linestyle="dashed", color="r", lw=0.8)
            plt.xlim(xlm)
            plt.ylabel("MV Voltage, pu")
            if sf:
                sff("mv_voltage_ts")
            tlps()


        if rd["plot_options"]["trn_powers"]:
            spri = np.array([np.abs(s.Spri) for s in slns])
            ssec = np.array([np.abs(s.Ssec) for s in slns])

            trn_kva = d.getObjAttr(d.TRN, val="kva")
            spri_rating = sum(trn_kva[:2])
            ssec_ratings = np.array([trn_kva[i - 1] for i in simulation.ckts.trnIdx])

            fig, [ax0, ax1] = plt.subplots(
                ncols=2,
                sharey=True,
            )
            ax0.plot(tt, 100 * spri / spri_rating)
            ax0.hlines(
                100,
                tt[0],
                tt[-1],
                linestyles="dashed",
            ),
            set_day_label(
                ax=ax0,
            )
            ax0.set_ylabel("Substation utilization, %")
            ax0.set_title("Primary Sub. Utilization")
            ax1.plot(
                tt,
                100 * ssec / ssec_ratings,
            )
            ax1.hlines(
                100,
                tt[0],
                tt[-1],
                linestyles="dashed",
            ),
            set_day_label(
                ax=ax1,
            )
            plt.legend(
                simulation.ckts.ldNo,
                fontsize="small",
                title="LV Ckt No.",
            )
            ax1.set_title("Secondary Sub. Utilization")
            ylm = plt.ylim()
            plt.ylim((min([ylm[0], -1]), max([ylm[1], 101])))
            tlps()


        if rd["plot_options"]["profile_options"]:
            # Plot each of the profiles which only has a dimension of 1. [The profiles
            # with '_' appended at the end are averages of other 2-d profiles.]
            fig, ax = plt.subplots(
                figsize=(
                    6.2,
                    3.6,
                )
            )
            ksel = [k for k, v in simulation.p.items() if v.ndim == 1]
            clrs = new_hsl_map(len(ksel), 100, 50)

            mrks = ["--", "-.", ":"] * (1 + (len(ksel) // 3))
            for k, clr, mrk in zip(ksel, clrs, mrks):
                plt.plot(tt, simulation.p[k], mrk, color=clr, label=k)

            set_day_label()
            plt.legend(title="Profile", fontsize="small", loc=(1.05, 0.1))
            plt.ylabel("Power (kW, or normalised)")
            if sf:
                sff(
                    "profile_options",
                )

            tlps()


        if rd["plot_options"]["pmry_powers"] or rd["plot_options"]["pmry_loadings"]:
            splt = np.array([np.abs(np.sum(ss["Sfmv"], axis=1)) for ss in slns])
            if rd["plot_options"]["pmry_loadings"]:
                yy = 1e2 * 1e-3 * splt / np.array([v for v in simulation.fdr2pwr.values()])  # in %
                plt.plot(tt, yy)
                _ = [
                    plt.text(0, yy[0][i], f"{b} (F{i+1}, {p} MVA)")
                    for i, (b, p) in enumerate(simulation.fdr2pwr.items())
                ]
                set_day_label()
                plt.ylabel("Power, % of rated")
                if sf:
                    sff("pmry_loadings")

                tlps()

            if rd["plot_options"]["pmry_powers"]:
                plt.plot(tt, splt / 1e3)
                _ = [
                    plt.text(0, splt[0][i] / 1e3, f"{b} (F{i+1}, {p} MVA)")
                    for i, (b, p) in enumerate(simulation.fdr2pwr.items())
                ]
                set_day_label()
                plt.ylabel("Power, MVA")
                if sf:
                    sff("pmry_powers")

                tlps()

        if rd["plot_options"]["profile_sel"][0]:
            # ksel_opts = [k for k,v in simulation.p.items() if v.ndim==2] # list the options
            ksel = rd["plot_options"]["profile_sel"][1]
            nplt = 10
            fig, [ax0, ax1] = plt.subplots(
                figsize=(8, 3.6),
                ncols=2,
                nrows=1,
                sharey=True,
                sharex=True,
            )
            clrs = new_hsl_map(nplt)

            _ = [
                ax0.plot(
                    tt,
                    simulation.p[ksel][i],
                    ".-",
                    color=clrs[i],
                )
                for i in range(nplt)
            ]
            ax0.set_title(f'Plotting {nplt} oo {len(simulation.p[ksel])} "{ksel}" profiles\n')
            ax0.set_ylabel("Power, kW")
            ax0.set_xlabel("Hour of the day")

            fillplot(
                simulation.p[ksel].T,
                tt,
                ax=ax1,
            )
            plt.plot(tt, np.median(simulation.p[ksel], axis=0), "k", label="Median")
            plt.plot(tt, np.mean(simulation.p[ksel], axis=0), "b--", label="Mean")
            ax1.set_title(f'Range, Quartiles, Median\nand Mean of "{ksel}" profiles')
            ax1.legend()
            set_day_label()
            if sf:
                sff(f"profile_sel_{ksel}")

            tlps()

if __name__ == "__main__":
    run_dss_simulation(aox.run_dict0)
