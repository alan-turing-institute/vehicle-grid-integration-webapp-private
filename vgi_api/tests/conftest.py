from pydantic.networks import import_email_validator
import pytest
import io


def create_file(data: str) -> io.BytesIO:

    file_like = io.BytesIO()
    file_like.write(data.encode())
    file_like.seek(0)

    return file_like


@pytest.fixture()
def valid_profile_csv():
    """Valid CSV profile data

    Header: [time, profile_1, profile_2, ..., profile_k]
        Can have any names as long as the first column is the time column
        Can have K profiles where K > 1

        time column format: 'HH:MM:SS'.
            Can be any values but must be at 30 minute intervals
            Must have 12 hours of data 24 data points
    """

    csv_data = (
        "Time, profile_1, profile_2, profile_3\n"
        "00:00:00, 0.1, 1.2, 0.3\n"
        "00:30:00, 1.1, 1.2, 0.3\n"
        "01:00:00, 12.1, 1.2, 0.3\n"
        "01:30:00, 0.1, 1.2, 0.3\n"
        "02:00:00, 0.1, 1.2, 0.3\n"
        "02:30:00, 23.1, 1.2, 0.3\n"
        "03:00:00, 0.3, 1.2, 0.3\n"
        "03:30:00, 0.1, 1.2, 0.3\n"
        "04:00:00, 0.1, 1.2, 0.3\n"
        "04:30:00, 0.1, 1.2, 0.3\n"
        "05:00:00, 0.1, 1.2, 0.3\n"
        "05:30:00, 0.1, 1.2, 0.3\n"
        "06:00:00, 0.1, 1.2, 0.3\n"
        "06:30:00, 0.1, 1.2, 0.3\n"
        "07:00:00, 0.1, 1.2, 0.3\n"
        "07:30:00, 0.1, 1.2, 0.3\n"
        "08:00:00, 0.1, 1.2, 0.3\n"
        "08:30:00, 0.1, 1.2, 0.3\n"
        "09:00:00, 0.1, 1.2, 0.3\n"
        "09:30:00, 0.1, 1.2, 0.3\n"
        "10:00:00, 0.1, 1.2, 0.3\n"
        "10:30:00, 0.1, 1.2, 0.3\n"
        "11:00:00, 0.1, 1.2, 0.3\n"
        "11:30:00, 10.2, 1.2, 11.2"
    )

    return create_file(csv_data)
