import json
import base64
import os
from pathlib import Path
from vgi_api import azure_mockup
from vgi_api import azureOptsXmpls as aox
import pytest


def simrun(n_lv=15, n_id=1060):
    n_lv = int(n_lv)
    n_id = int(n_id)

    parameters = aox.run_dict0
    parameters["network_data"]["n_id"] = n_id
    parameters["network_data"]["n_lv"] = n_lv

    # Based on example from LocalFunctionProj

    fig1, fig2 = azure_mockup.run_dss_simulation(parameters)

    resultdict = {
        "parameters": parameters,
        "plot1": base64.b64encode(fig1.getvalue()).decode("utf-8"),
        "plot2": base64.b64encode(fig2.getvalue()).decode("utf-8"),
    }

    json_string = json.dumps(resultdict)

    return json_string


def test_simrun():
    with open(os.path.join(Path(__file__).parent, "dsssimulation_5_1060.json")) as fp:
        dsssimulation_5_1060 = fp.read()
        ## compare the logical json, not the on disk representation
        ## formatting and line endings changes should not fail

        saved = json.loads(dsssimulation_5_1060)
        simulated = json.loads(simrun(5, 1060))

        assert saved["parameters"] == simulated["parameters"]
        assert saved["plot1"] == simulated["plot1"]
        assert saved["plot2"] == simulated["plot2"]
