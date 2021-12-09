from fastapi.testclient import TestClient
from devtools import debug
import pytest
from pydantic import ValidationError
from vgi_api import app
from vgi_api.validation import (
    NetworkID,
    VALID_LV_NETWORKS_URBAN,
    VALID_LV_NETWORKS_RURAL,
    DEFAULT_LV_NETWORKS,
    DefaultLV,
    ValidateLVParams,
)

client = TestClient(app)


@pytest.mark.parametrize(
    "params",
    [
        (
            dict(
                lv_list=",".join([str(i) for i in VALID_LV_NETWORKS_URBAN[:5]]),
                n_id=NetworkID.URBAN.value,
            )
        ),
        (
            dict(
                lv_default=DefaultLV.NEAR_SUB,
                n_id=NetworkID.URBAN.value,
            )
        ),
    ],
)
def test_valid_params_simulation(params):
    """Test we get 200 status code with valid EV Distribution Network
    parameters"""

    response = client.post(
        "/simulate",
        params=params,
    )

    debug(response.json())
    assert response.status_code == 200


@pytest.mark.parametrize(
    "params,loc",
    [
        (  # No n_id
            dict(
                lv_list=",".join([str(i) for i in VALID_LV_NETWORKS_URBAN[:5]]),
                # n_id=NetworkID.URBAN.value,
            ),
            ["query", "n_id"],
        ),
        (
            # Missing lv_list and lv_default
            dict(
                n_id=NetworkID.URBAN.value,
            ),
            ["lv_default"],
        ),
        (
            # Wrond default value
            dict(
                n_id=NetworkID.URBAN.value,
                lv_default=4,
            ),
            ["query", "lv_default"],
        ),
        (
            # xfmr_scale less than zero
            dict(
                lv_list=",".join([str(i) for i in VALID_LV_NETWORKS_URBAN[:5]]),
                n_id=NetworkID.URBAN.value,
                xfmr_scale=-0.3,
            ),
            ["query", "xfmr_scale"],
        ),
    ],
)
def test_invalid_params_simulation(params, loc):
    """Test we get 200 status code with valid EV Distribution Network
    parameters"""

    response = client.post(
        "/simulate",
        params=params,
    )

    payload = response.json()
    debug(payload)
    assert response.status_code == 422
    # Check the parameter that was rejected was the correct one
    assert payload["detail"][0]["loc"] == loc


@pytest.mark.parametrize(
    "lv_list,n_id, lv_default",
    [
        ("1101,1102", NetworkID.URBAN.value, None),
        (None, NetworkID.URBAN.value, DefaultLV.NEAR_SUB),
        ("1101,1102,1103,1104,1105", NetworkID.URBAN.value, None),
    ],
)
def test_success_validate_lv_params(lv_list, n_id, lv_default):

    ValidateLVParams(lv_list=lv_list, n_id=n_id, lv_default=lv_default)


@pytest.mark.parametrize(
    "lv_list,n_id, lv_default",
    [
        (None, NetworkID.URBAN.value, None),  # must pass one of lv_list of lv_default
        ("2222", NetworkID.URBAN.value, None),  # Not in the list of valid network ids
        ("", NetworkID.URBAN.value, None),
        ("1101,1102,1103,1104,1105,1106", NetworkID.URBAN.value, None),
    ],
)
def test_failure_validate_lv_params(lv_list, n_id, lv_default):

    with pytest.raises(ValidationError) as err:
        ValidateLVParams(lv_list=lv_list, n_id=n_id, lv_default=lv_default)
    debug(err)


@pytest.mark.skip(reason="Not implemented")
def test_upload_csv():

    pass
