from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

FEATURE_NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

class PatientFeatures(BaseModel):
    """
    Data Contract for Patient Features
    """
    age: float = Field(..., ge=0, le=120, description = "Age of the patient in years")
    sex: Literal[1,2] = Field(..., description = "Sex (1 = male, 2 = female)")
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body mass index (kg/m^2)")
    bp: float = Field(..., ge=40.0, le=200.0, description="Average blood pressure (MAP)")
    s1: float = Field(..., ge=50.0, le=400.0, description="Total serum cholesterol")
    s2: float = Field(..., ge=20.0, le=300.0, description="Low-density lipoproteins")
    s3: float = Field(..., ge=10.0, le=150.0, description="High-density lipoproteins")
    s4: float = Field(..., ge=1.0, le=15.0, description="Total cholesterol / HDL")
    s5: float = Field(..., ge=2.0, le=7.0, description="Log of serum triglycerides")
    s6: float = Field(..., ge=40.0, le=400.0, description="Blood sugar level")

    def to_numpy(self) -> np.ndarray:
        """Guarantees the output array matches exact training order."""
        return np.array([[getattr(self, col) for col in FEATURE_NAMES]])
    
class PredictionResponse(BaseModel):
    disease_progression_score: float = Field(..., description="Predicted progression score one year baseline")
    reference_range: str = Field(default="25 to 346", description="Expected baseline range per Phase 0 contract")