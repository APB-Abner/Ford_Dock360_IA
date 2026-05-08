import pytest
import pandas as pd
from src.pipeline.config import LEAKAGE_BEHAVIORAL
from src.pipeline.train_churn_real import check_leakage

def test_leakage_detection():
    # Test each behavioral column
    for col in LEAKAGE_BEHAVIORAL:
        df = pd.DataFrame({
            col: [1, 2, 3],
            "ano_modelo": [2020, 2021, 2022],
            "modelo": ["KA", "KA", "KA"]
        })
        with pytest.raises(ValueError, match=f"Colunas comportamentais \(leakage\) em X: \['{col}'\]"):
            check_leakage(df)

def test_no_leakage_passes():
    df = pd.DataFrame({
        "ano_modelo": [2020, 2021, 2022],
        "modelo": ["KA", "KA", "KA"],
        "dias_ate_entrega": [10, 15, 20]
    })
    # Should not raise any error
    check_leakage(df)

def test_leakage_count():
    assert len(LEAKAGE_BEHAVIORAL) == 11
