from pydantic import BaseModel, ConfigDict
from datetime import datetime
import numpy as np
import pandas as pd


class UserData(BaseModel):
    QC: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Loop(BaseModel):
    ID: int 
    start: datetime
    end: datetime
    coordinates: np.ndarray 
    BMU: tuple[int, ...] | None  = None
    distance: float | None  = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

