import numpy as np
from sklearn.datasets import load_diabetes
from diabetes_api.preprocessing import EfronScaler

def test_efron_scaler_matches():
    """
    Proves that our custom EfronScaler perfectly replicates the 
    opaque scaling applied in sklearn's load_diabetes(scaled=True).
    """

    raw_X = load_diabetes(scaled=False).data
    target_scaled_X = load_diabetes(scaled=True).data

    scaler = EfronScaler()
    custom_scaled_X = scaler.fit_transform(raw_X)

    np.testing.assert_allclose(
        custom_scaled_X,
        target_scaled_X,
        rtol=1e-5,
        atol=1e-8,
        err_msg="EfronScaler output does not match sklearn scaled data!"
    )