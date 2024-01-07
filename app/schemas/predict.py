from typing import Any, List, Optional
import sys
from pathlib import Path

from pydantic import BaseModel
from model.preprocessing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: int


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "has_gas": "t",
                        "origin_up": "kamkkxfxxuwbdslkwifmmcsiusiuosws",
                        "price_change_energy": "decrease",
                        "cons_12m": 4660,
                        "forecast_cons_12m": 189.95,
                        "forecast_discount_energy": 0.0,
                        "forecast_meter_rent_12m": 16.27,
                        "imp_cons": 0.0,
                        "margin_gross_pow_ele": 16.38,
                        "nb_prod_act": 1,
                        "net_margin": 18.89,
                        "pow_max": 13.8,
                        "price_off_peak_var": 0.149609,
                        "price_off_peak_fix": 44.311375,
                        "previous_price": 44.460984,
                        "price_sens": 0.960813,
                        "end_year": 2016,
                        "modif_prod_month": 8,
                        "renewal_year": 2015,
                        "renewal_month": 8,
                        "diff_act_end": 2566,
                        "diff_act_modif": 0,
                        "diff_end_modif": 2566,
                        "ratio_last_month_last12m_cons": 0.0,
                    }
                ]
            }
        }
