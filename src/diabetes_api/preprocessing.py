import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EfronScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler to match the exact transformation used in Efron et al. (2004)
    and scikit-learn's load_diabetes dataset(scaled = True)

    Transformation: (X - mean) / (std * dqrt(n_samples))
    Note: The scaling factor uses the population standard deviation (ddof=0).
    """

    def __init__(self):
        self.means_ = None
        self.scale_factors_ = None
        self.n_samples_ = None

    def fit(self, X, y=None):
        X_array = np.asarray(X)
        self.n_samples_ = X_array.shape[0]
        self.means_ = np.mean(X_array, axis=0)

        # std * sqrt(n_samples) is mathematically equivalent to the L2 norm of the centered vector
        # We use ddof=0 which is numpy's default for standard deviation

        stds = np.std(X_array, axis=0, ddof=0)
        self.scale_factors_ = stds * np.sqrt(self.n_samples_)

        # Guard against division by zero for constant features

        self.scale_factors_[self.scale_factors_ == 0.0] = 1.0

        return self

    def transform(self, X):
        if self.means_ is None or self.scale_factors_ is None:
            raise RuntimeError("EfronScaler has not been fitted yet.")

        X_array = np.asarray(X)
        return (X_array - self.means_) / self.scale_factors_
