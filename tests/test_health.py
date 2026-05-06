import os
import sys
from pathlib import Path
from unittest.mock import patch

# Mandato: SECRET_KEY deve ser setada antes do import da API
os.environ.setdefault('SECRET_KEY', 'x' * 32)

from fastapi.testclient import TestClient

# Root path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.main import app

client = TestClient(app)


def test_health_503_sem_modelos():
    """Verifica se o health check retorna 503 quando modelos nao existem."""
    # Patch do _MODELS_DIR no router de health para um diretorio inexistente
    with patch('src.api.routers.health._MODELS_DIR', Path('/tmp/nonexistent_models_test_dir')):
        response = client.get('/health')
    
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
