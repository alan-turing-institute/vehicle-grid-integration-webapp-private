"""Tests related to API argument validation"""
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
    MVSolarPVOptions,
    DefaultLV,
    ValidateLVParams,
)
import io
import requests


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
        app.url_path_for("simulate"),
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
        app.url_path_for("simulate"),
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


@pytest.mark.parametrize(
    "n_id",
    [(NetworkID.RURAL), (NetworkID.URBAN)],
)
def test_lv_network(n_id):

    response = client.get(app.url_path_for("lv_network"), params={"n_id": n_id.value})

    payload = response.json()
    debug(payload)

    assert response.status_code == 200

    if n_id == NetworkID.RURAL:
        assert payload["networks"] == VALID_LV_NETWORKS_RURAL
    elif n_id == NetworkID.URBAN:
        assert payload["networks"] == VALID_LV_NETWORKS_URBAN


@pytest.mark.parametrize(
    "n_id",
    [NetworkID.RURAL, NetworkID.URBAN],
)
@pytest.mark.parametrize(
    "lv_default",
    [DefaultLV.NEAR_SUB, DefaultLV.MIXED, DefaultLV.NEAR_EDGE],
)
def test_lv_network_defaults(n_id, lv_default):

    response = client.get(
        app.url_path_for("lv_network_defaults"),
        params={"n_id": n_id.value, "lv_default": lv_default.value},
    )

    payload = response.json()
    debug(payload)

    assert response.status_code == 200

    assert payload["networks"] == DEFAULT_LV_NETWORKS[n_id][lv_default]


def upload_csv(file: io.BytesIO) -> requests.Response:

    file_name = "example_profile.csv"
    upload_file = {"mv_solar_pv_csv": (file_name, file)}
    return client.post(
        app.url_path_for("simulate"),
        files=upload_file,
        params={
            "lv_default": DefaultLV.NEAR_SUB.value,
            "n_id": NetworkID.URBAN.value,
            "mv_solar_pv_profile": MVSolarPVOptions.CSV.value,
        },
    )


def test_valid_csv(valid_profile_csv: io.BytesIO):

    resp = upload_csv(valid_profile_csv)
    debug(resp.json())
    assert resp.status_code == 200


def test_invalid_csv_long(invalid_profile_csv_too_long: io.BytesIO):

    resp = upload_csv(invalid_profile_csv_too_long)
    debug(resp.json())
    assert resp.status_code == 422


def test_invalid_csv_short(invalid_profile_csv_too_short: io.BytesIO):

    resp = upload_csv(invalid_profile_csv_too_short)
    debug(resp.json())
    assert resp.status_code == 422


def test_invalid_csv_wrong_time(invalid_profile_wrong_time: io.BytesIO):

    resp = upload_csv(invalid_profile_wrong_time)
    debug(resp.json())
    assert resp.status_code == 422


def test_invalid_csv_offset(invalid_profile_csv_offset: io.BytesIO):

    resp = upload_csv(invalid_profile_csv_offset)
    debug(resp.json())
    assert resp.status_code == 422


def test_invalid_csv_not_float(invalid_profile_csv_not_float: io.BytesIO):

    resp = upload_csv(invalid_profile_csv_not_float)
    debug(resp.json())
    assert resp.status_code == 422
