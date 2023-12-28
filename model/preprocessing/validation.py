import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = json.loads(error.json())

    return validated_data, errors


class DataInputSchema(BaseModel):
    has_gas: Optional[str]
    origin_up: Optional[str]
    price_change_energy: Optional[str]
    cons_12m: Optional[int]
    forecast_cons_12m: Optional[float]
    forecast_discount_energy: Optional[float]
    forecast_meter_rent_12m: Optional[float]
    imp_cons: Optional[float]
    margin_gross_pow_ele: Optional[float]
    nb_prod_act: Optional[int]
    net_margin: Optional[float]
    pow_max: Optional[float]
    price_off_peak_var: Optional[float]
    price_off_peak_fix: Optional[float]
    previous_price: Optional[float]
    price_sens: Optional[float]
    end_year: Optional[int]
    modif_prod_month: Optional[int]
    renewal_year: Optional[int]
    renewal_month: Optional[int]
    diff_act_end: Optional[int]
    diff_act_modif: Optional[int]
    diff_end_modif: Optional[int]
    ratio_last_month_last12m_cons: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
