import json
import base64
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


@pytest.mark.xfail(reason="I'm not sure when this last passed")
def test_simrun():
    with open("tests/dsssimulation_15_1060.json") as fp:
        dsssimulation_15_1060 = fp.read()
        ## compare the logical json, not the on disk representation
        ## formatting and line endings changes should not fail
        assert json.loads(simrun(15, 1060)) == json.loads(dsssimulation_15_1060)
