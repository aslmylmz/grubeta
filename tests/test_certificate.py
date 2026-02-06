import pytest
import json
import sys
import numpy as np
import pandas as pd
from grubeta.temporal_integrity import TemporalCertificate

def test_certificate_metadata():
    cert = TemporalCertificate(
        model_version="1.0.0",
        training_window=("2020-01-01", "2020-06-01"),
        prediction_window=("2020-06-02", "2020-12-31")
    )
    
    # Check sustenance metadata
    assert cert.python_version == sys.version.split()[0]
    assert cert.numpy_version == np.__version__
    assert cert.pandas_version == pd.__version__

def test_certificate_export():
    cert = TemporalCertificate(
        model_version="1.0.0",
        training_window=("A", "B"),
        prediction_window=("C", "D"),
        integrity_checks_passed=["CHECK_1", "CHECK_2"]
    )
    
    # JSON Check
    json_str = cert.to_json()
    data = json.loads(json_str)
    assert data["model_version"] == "1.0.0"
    assert data["environment"]["python"] == sys.version.split()[0]
    assert "CHECK_1" in data["integrity_checks_passed"]
    
    # Text Check
    text = str(cert)
    assert "TEMPORAL INTEGRITY CERTIFICATE" in text
    assert "Environment: Python" in text
    assert "  ✓ CHECK_1" in text
