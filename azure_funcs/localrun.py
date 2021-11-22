import logging
import json
import base64
from dsssimulation import azure_mockup
from dsssimulation import azureOptsXmpls as aox
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import azure.functions as func

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

n_lv = 15
n_lv = int(n_lv)

n_id = 1060
n_id = int(n_id)

parameters = aox.run_dict0
parameters["network_data"]["n_id"] = n_id
parameters["network_data"]["n_lv"] = n_lv

# Based on example from LocalFunctionProj

fig1, fig2 = azure_mockup.run_dss_simulation(parameters)

logging.info("Package results")

resultdict = {
    "parameters": parameters,
    "plot1": base64.b64encode(fig1.getvalue()).decode("utf-8"),
    "plot2": base64.b64encode(fig2.getvalue()).decode("utf-8"),
}

json_string = json.dumps(resultdict)

logging.info("Results packaged, now return")
