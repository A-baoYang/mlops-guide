from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

import pandas as pd
from pydantic import BaseModel

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner])


class IrisFeatures(BaseModel):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float

    # Optional field
    request_id: Optional[int]

    # Use custom Pydantic config for additional validation options
    class Config:
        extra = "forbid"


input_spec = JSON(pydantic_model=IrisFeatures)


@svc.api(input=input_spec, output=NumpyNdarray())
def classify(input_data: IrisFeatures) -> NDArray[Any]:
    if input_data.request_id is not None:
        print("Received request ID: ", input_data.request_id)

    input_df = pd.DataFrame([input_data.dict(exclude={"request_id"})])
    print(input_df)
    return iris_clf_runner.run(input_df)
