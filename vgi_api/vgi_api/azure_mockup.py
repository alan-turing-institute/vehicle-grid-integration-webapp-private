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
from .funcsPython_turing import gDir, fillplot, set_day_label, new_hsl_map
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import tempfile
import numpy as np
import logging

fn_root = sys.path[0] if __name__ == "__main__" else os.path.dirname(__file__)


def run_dss_simulation(rd=aox.run_dict0, sf=0):

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
        simulation = ft.turingNet(frId=1000, frId0=frid0, rundict=rd, mod_dir=temp_dir)

    # tsp_n = rd["simulation_data"]["ts_profiles"]["n"]
    tt = np.arange(0, 24, 24 / 48)  # get the clock

    # Get the solutions
    lds = simulation.get_lds_kva(48)
    slns, _ = simulation.run_dss_lds(lds)

    # From here: create all plots
    plt.close("all")

    # PLOT: mv_highlevel, network plot highlighting LV networks etc
    plt.clf()
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
    mv_highlevel_buffer = io.BytesIO()
    plt.gcf().savefig(mv_highlevel_buffer, facecolor="DarkGray")

    # PLOT: mv_highlevel_clean, network plot without further highlights
    plt.clf()
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
    mv_highlevel_clean_buffer = io.BytesIO()
    plt.gcf().savefig(mv_highlevel_clean_buffer, facecolor="DarkGray")

    # PLOT: mv_powers, power plot (no LV circles)
    plt.clf()
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
    mv_powers_buffer = io.BytesIO()
    plt.gcf().savefig(mv_powers_buffer, facecolor="LightGray")

    # PLOT: lv_voltages, compare two lv voltage timeseries
    plt.clf()
    assert (
        len(rd["plot_options"]["lv_voltages"]) < 3
    )  # for now only compare two profiles

    fig, ax = plt.subplots()
    lv_idxs = [
        simulation.ckts.ldNo.index(nn) for nn in rd["plot_options"]["lv_voltages"]
    ]

    clrs = [
        "C0",
        "C3",
    ]
    for idx, clr in zip(lv_idxs, clrs):
        Vsec = np.array([s["VlvLds"][idx] for s in slns])[:, :, 0]
        # ToDo: Return data in csv file
        _, dplt = fillplot(
            np.abs(Vsec) / 230,
            tt,
            ax=ax,
            lineClrs=[clr],
            fillKwargs={"color": clr},
        )

    _ = [
        plt.plot(np.nan, np.nan, color=clr, label=lbl)
        for lbl, clr in zip(rd["plot_options"]["lv_voltages"], clrs)
    ]
    plt.legend(
        title="LV Network ID",
        fontsize="small",
    )
    set_day_label()
    plt.ylabel("Voltage, pu (230 V base)")
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

    lv_voltages_buffer = io.BytesIO()
    plt.gcf().savefig(lv_voltages_buffer, facecolor="LightGray")

    # PLOT: lv_comparison
    plt.clf()
    fig, axs = plt.subplots(
        figsize=(
            9,
            3.2,
        ),
        nrows=1,
        ncols=simulation.ckts.N,
        sharey=True,
        sharex=True,
    )

    data_out_lv_comparison = np.zeros(
        (
            48,
            0,
        )
    )
    for ii, ax in enumerate(axs):
        Vsec = np.array([s["VlvLds"][ii] for s in slns])[:, :, 0]
        _, dplt = fillplot(
            np.abs(Vsec) / 230,
            tt,
            ax=ax,
            lineClrs=[
                f"C{ii}",
            ],
            fillKwargs={"color": f"C{ii}"},
        )
        xlm = plt.xlim()
        ax.hlines(
            [0.94, 1.10],
            *xlm,
            linestyles="dashed",
            color="r",
        )
        ax.set_title(
            f"LV Network: {simulation.ckts.ldNo[ii]}",
            fontsize="medium",
        )
        set_day_label()

        data_out_lv_comparison = np.c_[data_out_lv_comparison, dplt.T]

    head_lv_comparison = [
        f"{int(q)}% quantile: LV Network: {simulation.ckts.ldNo[ii]}"
        for ii in range(simulation.ckts.N)
        for q in np.linspace(0, 100, 5)
    ]

    axs[0].text(
        0.1,
        1.0905,
        "$Upper\;limit$",
        fontsize="small",
    )
    axs[0].text(
        0.1,
        0.9425,
        "$Lower\;limit$",
        fontsize="small",
    )
    axs[0].set_ylabel("Voltage, per unit")

    # Set table on second subplot, unless not enough plots. Then use the last subplot
    if len(axs) < 3:
        axs[len(axs) - 1].set_xlabel("Hour of the day")
    else:
        axs[2].set_xlabel("Hour of the day")
        axs[-1].set_xlabel("")

    xlm = plt.xlim()
    plt.xlim(xlm)
    plt.tight_layout()
    if sf:
        sff("lv_comparison")

    lv_comparison_buffer = io.BytesIO()
    plt.gcf().savefig(lv_comparison_buffer, facecolor="LightGray")

    # PLOT: mv_voltages, voltage plot against time
    plt.clf()
    smv2pu = lambda s: np.abs(s.Vmv) / simulation.vKvbase[simulation.mvIdx]
    vb = np.array([smv2pu(s) for s in slns])
    _, dplt = fillplot(vb, np.linspace(0, 24, 48))
    set_day_label()
    xlm = plt.xlim()
    plt.hlines([0.94, 1.06], *xlm, linestyle="dashed", color="r", lw=0.8)
    plt.xlim(xlm)
    plt.ylabel("MV Voltage, pu")
    plt.text(
        0.1,
        1.0535,
        "$Upper\;limit$",
    )
    plt.text(
        0.1,
        0.941,
        "$Lower\;limit$",
    )
    if sf:
        sff("mv_voltage_ts")

    mv_voltages_buffer = io.BytesIO()
    plt.gcf().savefig(mv_voltages_buffer, facecolor="LightGray")

    head_mv_voltages = [f"MV voltage: {qq}% quantile" for qq in np.linspace(0, 100, 5)]
    data_out_mv_voltage = dplt.T

    # PLOT: trn_powers
    plt.clf()
    spri = np.array([np.abs(s.Spri) for s in slns])
    ssec = np.array([np.abs(s.Ssec) for s in slns])

    trn_kva = d.getObjAttr(d.TRN, val="kva")
    spri_rating = sum(trn_kva[:2])
    ssec_ratings = np.array([trn_kva[i - 1] for i in simulation.ckts.trnIdx])

    fig, [ax0, ax1] = plt.subplots(
        ncols=2,
        sharey=True,
    )
    ax0.plot(tt, 100 * spri / spri_rating, ".-")
    ax0.hlines(
        100,
        tt[0],
        tt[-1] + 3,
        linestyles="dashed",
        color="r",
    ),
    set_day_label(
        ax=ax0,
    )
    ax0.set_ylabel("Substation utilization, %")
    ax0.set_title("Primary Sub. Utilization")
    ax1.plot(tt, 100 * ssec / ssec_ratings, ".-")
    ax1.hlines(
        100,
        tt[0],
        tt[-1] + 3,
        linestyles="dashed",
        color="r",
    ),
    set_day_label(
        ax=ax1,
    )
    lgns = [
        simulation.ckts.ldNo[i] + f" ({int(ssec_ratings[i])} kVA)"
        for i in range(simulation.ckts.N)
    ]
    plt.legend(
        lgns,
        fontsize="small",
        title="LV Network (rating)",
    )
    ax1.set_title("Secondary Sub. Utilization")
    ylm = plt.ylim()
    plt.ylim((min([ylm[0], -1]), max([ylm[1], 101])))
    if sf:
        sff("trn_powers")

    trn_powers_buffer = io.BytesIO()
    plt.gcf().savefig(trn_powers_buffer, facecolor="LightGray")

    head_trn_powers = ["Prmy. Sub. Util."] + ["Sdry. Sub. Util." + l for l in lgns]
    data_out_trn_powers = np.c_[
        np.expand_dims(100 * spri / spri_rating, axis=1), 100 * ssec / ssec_ratings
    ]

    # PLOT: profile_options
    # Plot each of the profiles which only has a dimension of 1. [The profiles
    # with '_' appended at the end are averages of other 2-d profiles.]
    plt.clf()
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
    plt.legend(title="Profile ID", fontsize="small", loc=(1.05, 0.1))
    plt.ylabel("Power, kW)")
    plt.tight_layout()
    if sf:
        sff(
            "profile_options",
        )

    profile_options_buffer = io.BytesIO()
    plt.gcf().savefig(profile_options_buffer, facecolor="LightGray")

    # Needed for pmry_powers and pmry_loadings plots
    splt = np.array([np.abs(np.sum(ss["Sfmv"], axis=1)) for ss in slns])

    # PLOT: pmry_loadings
    plt.clf()
    yy = 1e2 * 1e-3 * splt / np.array([v for v in simulation.fdr2pwr.values()])  # in %
    _ = [
        plt.plot(tt, yy[:, i], color=matplotlib.cm.tab20(i)) for i in range(yy.shape[1])
    ]
    lgnd = [
        f"F{i+1} (to {b}), {p} MVA"
        for i, (b, p) in enumerate(simulation.fdr2pwr.items())
    ]
    plt.legend(lgnd, loc=(1.03, 0.2), fontsize="small", title="Feeder (to), rating")
    plt.hlines(
        100,
        tt[0],
        tt[-1] + 3,
        linestyles="dashed",
        color="r",
    )
    set_day_label()
    plt.ylabel("Power, % of rated")
    plt.tight_layout()
    if sf:
        sff("pmry_loadings")

    pmry_loadings_buffer = io.BytesIO()
    plt.gcf().savefig(pmry_loadings_buffer, facecolor="LightGray")

    # Todo: Return from API
    head_primary_loadings = lgnd
    data_out_primary_loadings = yy

    # PLOT: pmry_powers
    plt.clf()
    plt.plot(tt, splt / 1e3)
    _ = [
        plt.text(0, splt[0][i] / 1e3, f"{b} (F{i+1}, {p} MVA)")
        for i, (b, p) in enumerate(simulation.fdr2pwr.items())
    ]
    set_day_label()
    plt.ylabel("Power, MVA")
    plt.tight_layout()
    if sf:
        sff("pmry_powers")

    pmry_powers_buffer = io.BytesIO()
    plt.gcf().savefig(pmry_powers_buffer, facecolor="LightGray")

    # # PLOT: profile_sel
    # plt.clf()
    # # ksel_opts = [k for k,v in simulation.p.items() if v.ndim==2] # list the options
    # ksel = rd["plot_options"]["profile_sel"][0]
    # nplt = 10
    # fig, [ax0, ax1] = plt.subplots(
    #     figsize=(8, 3.6),
    #     ncols=2,
    #     nrows=1,
    #     sharey=True,
    #     sharex=True,
    # )
    # clrs = new_hsl_map(nplt)

    # _ = [
    #     ax0.plot(
    #         tt,
    #         simulation.p[ksel][i],
    #         ".-",
    #         color=clrs[i],
    #     )
    #     for i in range(nplt)
    # ]
    # ax0.set_title(
    #     f'Plotting {nplt} of {len(simulation.p[ksel])} "{ksel}" profiles\n'
    # )
    # ax0.set_ylabel("Power, kW")
    # ax0.set_xlabel("Hour of the day")

    # _, dplt = fillplot(
    #     simulation.p[ksel].T,
    #     tt,
    #     ax=ax1,
    # )
    # plt.plot(tt, np.median(simulation.p[ksel], axis=0), "k", label="Median")
    # plt.plot(tt, np.mean(simulation.p[ksel], axis=0), "b--", label="Mean")
    # ax1.set_title(f'Range, Quartiles, Median\nand Mean of "{ksel}" profiles')
    # ax1.legend()
    # set_day_label()
    # plt.tight_layout()
    # if sf:
    #     sff(f"profile_sel_{ksel}")

    # profile_sel_buffer = io.BytesIO()
    # plt.gcf().savefig(profile_sel_buffer, facecolor="LightGray")

    return (
        mv_highlevel_buffer,
        lv_voltages_buffer,
        lv_comparison_buffer,
        mv_voltages_buffer,
        mv_powers_buffer,
        mv_highlevel_buffer,
        mv_highlevel_clean_buffer,
        trn_powers_buffer,
        profile_options_buffer,
        pmry_loadings_buffer,
        pmry_powers_buffer,
        head_primary_loadings,
        data_out_primary_loadings,
        head_mv_voltages,
        data_out_mv_voltage,
        head_trn_powers,
        data_out_trn_powers,
        head_lv_comparison,
        data_out_lv_comparison,
    )


if __name__ == "__main__":
    run_dss_simulation(aox.run_dict0)
