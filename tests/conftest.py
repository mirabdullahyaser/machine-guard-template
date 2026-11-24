"""Shared test fixtures: small, fast pieces of data that every test file can ask for by name."""
import pandas as pd
import pytest

from machine_guard.features import RAW_FEATURES, load_data

# One healthy-looking run, in the API's request format (the course contract example).
HEALTHY_RUN = {
    "type": "L",
    "air_temp_k": 298.1,
    "process_temp_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    """The full training file, loaded once for the whole test session."""
    return load_data("data/raw/ai4i2020.csv")


@pytest.fixture
def sample(data) -> pd.DataFrame:
    """200 clean rows, copied so a test can break them without affecting other tests."""
    return data.head(200).copy()


@pytest.fixture
def healthy_run() -> dict:
    return dict(HEALTHY_RUN)


@pytest.fixture
def healthy_frame(healthy_run) -> pd.DataFrame:
    return pd.DataFrame([healthy_run])[RAW_FEATURES]
