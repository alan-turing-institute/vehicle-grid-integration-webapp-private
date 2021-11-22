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

fn_root = sys.path[0] if __name__=='__main__' else os.path.dirname(__file__)

def run_dss_simulation(rd, sf=0):

  logging.info("Entering run_dss_simulation")

  # Set up a temporary directory to store network files
  with tempfile.TemporaryDirectory() as temp_dir:
    ft.unzip_networks(dest_dir=os.path.join(temp_dir, "_network_mod"), n_id = rd['network_data']['n_id'])

    d = funcsDss_turing.dssIfc(dss.DSS)
    # Place modified files into _network_mod to match hardcoded value in slesNtwk_turing.py
    ntwk = ft.modify_network(rd, mod_dir=temp_dir, dnout="_network_mod")

    # Simulation modifications

    frid0 = rd['network_data']['n_id']
    simulation = ft.turingNet(frId=1000, frId0=frid0, rundict=rd, 
                              mod_dir=temp_dir)

    # Get the solutions
    tsp_n = rd['simulation_data']['ts_profiles']['n']
    lds = simulation.get_lds_kva(tsp_n)
    slns, _ = simulation.run_dss_lds(lds)

    # Plot the network we have created.
    simulation.fPrm.update({'saveFig': sf,
                            'sd': gDir,
                            'showFig': False,
                            'pdf': False,
                            'figname':' pltNetworks_mvonly_new'})
    simulation.plotXvNetwork(pType='B', pnkw={'txtOpts':'all'},)
    network_buffer = io.BytesIO()
    plt.gcf().savefig(network_buffer, facecolor="DarkGray")

    # Power plot (no LV circles)
    simulation.fPrm.update({'saveFig': sf,
                            'sd': os.path.join(fn_root,'data','mvlv_topos'),
                            'showFig': False,
                            'pdf': False,
                            'figname': f'{simulation.fdrs[frid0]}'})
    txtFss = {1060:'10', 1061:'6',}
    pnkw = {'txtOpts': 'all', 'lvnFlag': False, 'txtFs': txtFss[frid0],}
    simulation.plotXvNetwork(pType='p', pnkw=pnkw,)
    power_buffer = io.BytesIO()
    plt.gcf().savefig(power_buffer, facecolor="LightGray")

    # Voltage plot against time
    smv2pu = lambda s: np.abs(s.Vmv)/simulation.vKvbase[simulation.mvIdx]
    vb = np.array([smv2pu(s) for s in slns])
    fillplot(vb,np.linspace(0,24,tsp_n))
    set_day_label()
    xlm = plt.xlim()
    plt.hlines([0.94,1.06],*xlm,linestyle='dashed',color='r',lw=0.8)
    plt.xlim(xlm)
    plt.ylabel('MV Voltage, pu')
    plt.tight_layout()
    voltage_buffer = io.BytesIO()
    plt.gcf().savefig(voltage_buffer, facecolor="LightGray")
    # plt.show()

  return network_buffer, voltage_buffer


if __name__ == "__main__":
  run_dss_simulation(aox.run_dict0)