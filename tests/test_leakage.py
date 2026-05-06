import sys
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.config import LEAKAGE_COLUMNS
from src.pipeline.train_churn import check_leakage


def test_bloqueia_coluna_proibida():
    x = pd.DataFrame({"fez_primeira_revisao_rede": [1], "idade": [30]})
    with pytest.raises(ValueError):
        check_leakage(x)


def test_passa_features_limpas():
    x = pd.DataFrame({"idade": [30], "renda": [5000], "modelo_veiculo": ["ranger"]})
    check_leakage(x)


def test_len_leakage_columns():
    assert len(LEAKAGE_COLUMNS) == 12
