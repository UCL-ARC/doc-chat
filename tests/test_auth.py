"""Tests for authentication functionality."""

from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from src.participatory_ai_for_workshops.database import Base, get_db
from src.participatory_ai_for_workshops.main import app

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://test:test@localhost/test_db"

# Create test engine
engine_test = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=NullPool,
)
TestingSessionLocal = sessionmaker(
    engine_test,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Override get_db dependency for testing.

    Yields:
        AsyncSession: Test database session.

    """
    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@pytest.fixture(autouse=True)
async def setup_database() -> AsyncGenerator[None, None]:
    """
    Set up test database.

    Yields:
        None: None.

    """
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine_test.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def client() -> TestClient:
    """
    Create test client.

    Returns:
        TestClient: Test client instance.

    """
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


@pytest.fixture
def test_user() -> dict[str, str]:
    """
    Create test user data.

    Returns:
        Dict[str, str]: Test user data.

    """
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
    }


def test_signup(client: TestClient, test_user: dict[str, str]) -> None:
    """
    Test user signup.

    Args:
        client: Test client.
        test_user: Test user data.

    """
    response = client.post("/auth/signup", json=test_user)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user["email"]
    assert data["full_name"] == test_user["full_name"]
    assert "id" in data
    assert "password" not in data


def test_login(client: TestClient, test_user: dict[str, str]) -> None:
    """
    Test user login.

    Args:
        client: Test client.
        test_user: Test user data.

    """
    # First signup
    client.post("/auth/signup", json=test_user)

    # Then login
    response = client.post(
        "/auth/token",
        data={"username": test_user["email"], "password": test_user["password"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_credentials(
    client: TestClient, test_user: dict[str, str]
) -> None:
    """
    Test login with invalid credentials.

    Args:
        client: Test client.
        test_user: Test user data.

    """
    response = client.post(
        "/auth/token",
        data={"username": test_user["email"], "password": "wrongpassword"},
    )
    assert response.status_code == 401
