from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from model.config.core import config
from model.preprocessing.data_manager import load_dataset


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        load_dataset(
            client_file_name=config.app_config.client_data_file,
            price_file_name=config.app_config.price_data_file,
        ).iloc[0]
    ).T


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
