"""What a request and a response look like. Pydantic checks every request against these rules.

The limits come from machine_guard.features, the same place Day 4's data validation reads them.
So the API and the data checks can never disagree about what a sensible reading is.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from machine_guard.features import TYPE_VALUES, VALID_RANGES


def in_range(column: str, example: float):
    """A Field that only accepts values inside VALID_RANGES[column] (ends included)."""
    low, high = VALID_RANGES[column]
    return Field(..., ge=low, le=high, examples=[example],
                 description=f"Valid range {low} to {high}")


class SensorReading(BaseModel):
    """One machine run: the six raw sensor columns the model was trained on."""

    # Literal[...] means "only these exact values". It also shows a dropdown in /docs.
    type: Literal[tuple(TYPE_VALUES)] = Field(..., examples=["L"], description="Product quality: L, M or H")
    air_temp_k: float = in_range("air_temp_k", 298.1)

    # TODO 4: the four fields below accept ANY number, so a process temperature in Celsius (35.6) gets
    # through. Give each one a range check, the same way as air_temp_k above. Example values:
    # process_temp_k 308.6, rotational_speed_rpm 1551, torque_nm 42.8, tool_wear_min 0.
    process_temp_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float


class Prediction(BaseModel):
    # Pydantic reserves names starting with "model_"; this line allows our model_version field.
    model_config = ConfigDict(protected_namespaces=())

    failure_probability: float
    prediction: int
    threshold: float
    model_version: str


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
