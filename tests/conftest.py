import os

os.environ.setdefault("SECRET_KEY", "x" * 32)

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.security.auth import create_access_token


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def analyst_token():
    return create_access_token(subject="test-analyst", role="analyst")


@pytest.fixture(scope="session")
def viewer_token():
    return create_access_token(subject="test-viewer", role="viewer")


@pytest.fixture(scope="session")
def admin_token():
    return create_access_token(subject="test-admin", role="admin")


@pytest.fixture(scope="session")
def analyst_headers(analyst_token):
    return {"Authorization": f"Bearer {analyst_token}"}


@pytest.fixture(scope="session")
def viewer_headers(viewer_token):
    return {"Authorization": f"Bearer {viewer_token}"}


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
