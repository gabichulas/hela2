from sqlmodel import SQLModel, Field
from typing import Optional

class DistributionCenter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    lat: float
    lon: float
    description: Optional[str] = None
    
