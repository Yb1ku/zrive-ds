from zrive.module1 import get_data_meteo_api, prepare_data_to_plot, plot_data, VARIABLES
from unittest.mock import patch
import pytest
import os


url = "https://archive-api.open-meteo.com/v1/archive"


@pytest.fixture(scope="module")
def mock_api_response():
    def _build_response(status_code, with_hourly=True):
        common_data = {
            "latitude": 1.1,
            "longitude": 2.2,
            "generationtime_ms": 0.123,
            "utc_offset_seconds": 0,
            "timezone": "GMT",
            "timezone_abbreviation": "GMT",
            "elevation": 10.0,
        }
        if with_hourly:
            common_data["hourly"] = {
                "time": ["2010-01-01T00:00", "2010-01-01T01:00", "2010-01-01T02:00"],
                VARIABLES[0]: [1.0, 2.0, 3.0],
                VARIABLES[1]: [0.0, 0.5, 1.0],
                VARIABLES[2]: [5.0, 10.0, 15.0],
            }
        else:
            common_data["hourly"] = {}

        headers = {
            "Date": "Mon, 01 Jan 2020 00:00:00 GMT",
            "Content-Type": "application/json",
            "charset": "utf-8",
            "Transfer-Encoding": "chunked",
            "Connection": "keep-alive",
            "Content-Encoding": "deflate",
        }

        full_response = (common_data, status_code, headers)
        return full_response

    return _build_response


@pytest.fixture(scope="class")
def mock_response(request, mock_api_response):
    status = getattr(request.cls, "STATUS_CODE", 200)
    with patch(
        "zrive.module1.connect_to_api", return_value=mock_api_response(status)
    ) as mock_connect:
        request.cls.mock_connect = mock_connect
        yield


@pytest.fixture(scope="class")
def mock_data_for_plotting(request):
    data = {
        "time": ["2010-01-01T00:00", "2010-01-01T01:00", "2010-01-01T02:00"],
        VARIABLES[0]: [1.0, 2.0, 3.0],
        VARIABLES[1]: [0.0, 0.5, 1.0],
        VARIABLES[2]: [5.0, 10.0, 15.0],
    }
    request.cls.data = data


@pytest.mark.usefixtures("mock_response")
class TestGetDataMeteoAPI200:
    STATUS_CODE = 200

    def test_data_lengths(self):
        data = get_data_meteo_api("fake_url", "Madrid")
        lengths = [len(v) for v in data.values()]
        assert all(length == lengths[0] for length in lengths)

    def test_data_types(self):
        data = get_data_meteo_api("fake_url", "Madrid")
        assert isinstance(data["time"], list)
        assert all(isinstance(t, str) for t in data["time"])
        for var in VARIABLES:
            assert isinstance(data[var], list)
            assert all(isinstance(v, (int, float)) for v in data[var])

    def test_data_not_empty(self):
        data = get_data_meteo_api("fake_url", "Madrid")
        assert data, "Data should not be empty for status code 200"


@pytest.fixture(scope="class")
def mock_data_hourly(request):
    data = {
        "time": ["2010-01-01T00:00", "2010-01-01T01:00", "2010-01-01T02:00"],
        VARIABLES[0]: [1.0, 2.0, 3.0],
        VARIABLES[1]: [0.0, 0.5, 1.0],
        VARIABLES[2]: [5.0, 10.0, 15.0],
    }
    request.cls.data = data


@pytest.mark.usefixtures("mock_data_hourly")
class TestPrepareDataToPlot:
    def test_dataframe_index(self):
        df = prepare_data_to_plot(self.data)
        assert df.index.dtype == "object"
        assert all(isinstance(idx, str) for idx in df.index)

    def test_dataframe_non_empty(self):
        df = prepare_data_to_plot(self.data)
        assert not df.empty, "DataFrame should not be empty"


@pytest.mark.usefixtures("mock_data_for_plotting")
class TestPlotData:
    def test_plot_saves_file(self):
        data = prepare_data_to_plot(self.data)
        plot_data(data, "Madrid")
        assert os.path.exists("src/zrive/outputs/plot.png"), "Plot file was not created"
        output_path = "src/zrive/outputs/plot.png"
        os.remove(output_path)
