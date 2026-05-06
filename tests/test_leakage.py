import sys
from pathlib import Path

import pytest
import pandas as pd

# Root path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.config import LEAKAGE_COLUMNS
from src.pipeline.train_churn import check_leakage


def test_bloqueia_coluna_proibida():
    """Verifica se colunas proibidas disparam ValueError."""
    for col in LEAKAGE_COLUMNS:
        x = pd.DataFrame({col: [1], "idade": [30]})
        with pytest.raises(ValueError):
            check_leakage(x)


def test_passa_features_limpas():
    """Verifica se features permitidas passam sem erro."""
    x = pd.DataFrame({"idade": [30], "renda": [5000], "modelo_veiculo": ["ranger"]})
    check_leakage(x)


def test_len_leakage_columns():
    """Valida se a lista de leakage tem os 12 itens obrigatorios."""
    assert len(LEAKAGE_COLUMNS) == 12
