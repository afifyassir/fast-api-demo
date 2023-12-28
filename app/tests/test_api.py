from typing import Any, Union

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    def convert_timestamps(obj: Any) -> Union[str, Any]:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    test_data = test_data.applymap(convert_timestamps)

    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predictions"] in [0, 1]
    assert prediction_data["errors"] is None
