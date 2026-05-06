import os
import sys

os.environ.setdefault('SECRET_KEY', 'x' * 32)

from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.main import app

client = TestClient(app)


def test_health_503_sem_modelos():
    with patch('src.api.routers.health._MODELS_DIR', Path('/tmp/nonexistent_models_test_dir')):
        response = client.get('/health')
    assert response.status_code == 503
