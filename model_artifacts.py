"""Dependency guard for trusted local pipeline artifacts (never untrusted uploads)."""
import warnings

import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning


def load_trusted_pipeline(path, metadata):
    """Reject unsupported sklearn versions before deserializing executable joblib data."""
    if metadata.get("versions", {}).get("sklearn") != sklearn.__version__:
        raise ValueError("Artifact scikit-learn version is missing or incompatible; use its training environment or retrain.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except InconsistentVersionWarning:
            raise ValueError("Serialized estimator version is incompatible with the current environment.") from None
