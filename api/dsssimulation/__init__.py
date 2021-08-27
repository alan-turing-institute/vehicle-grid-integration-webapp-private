import logging
import json
import base64
from . import azure_mockup
from . import azureOptsXmpls as aox

import azure.functions as func
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    n_lv = req.params.get("n_lv")
    if not n_lv:
        n_lv = 5
    n_lv = int(n_lv)

    n_id = req.params.get("n_id")
    if not n_id:
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

    func.HttpResponse.mimetype = "application/json"
    func.HttpResponse.charset = "utf-8"

    logging.info("Results packaged, now return")

    return func.HttpResponse(json_string, status_code=200)

if __name__ == "__main__":
    main()
