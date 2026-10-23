import os

import pytest

# Tests never touch the real usage ledger; each TestClient lifespan gets a fresh in-memory one.
os.environ["USAGE_DB_PATH"] = ":memory:"


@pytest.fixture
def anyio_backend():
    return "asyncio"
