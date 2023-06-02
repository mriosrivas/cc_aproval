from typing import Any, List, Optional

from pydantic import BaseModel
from classification_model.processing.validation import UserInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleUserDataInputs(BaseModel):
    inputs: List[UserInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "amt_income_total": 76500,
                        "name_family_status": "widow",
                        "age":-20145,
                        "years_employed": -11907,
                        "occupation_type": "sales_staff"
                    }
                ]
            }
        }
