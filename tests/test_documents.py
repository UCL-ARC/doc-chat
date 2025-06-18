"""Tests for document functionality."""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def auth_headers(client: TestClient, test_user: dict[str, str]) -> dict[str, str]:
    """
    Create authentication headers.

    Args:
        client: Test client.
        test_user: Test user data.

    Returns:
        Dict[str, str]: Authentication headers.

    """
    # Signup and login
    client.post("/auth/signup", json=test_user)
    response = client.post(
        "/auth/token",
        data={"username": test_user["email"], "password": test_user["password"]},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_upload_document(client: TestClient, auth_headers: dict[str, str]) -> None:
    """
    Test document upload.

    Args:
        client: Test client.
        auth_headers: Authentication headers.

    """
    # Create a test PDF file
    test_file_path = "test.pdf"
    with open(test_file_path, "wb") as f:
        f.write(b"%PDF-1.4\n%Test PDF file")

    try:
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                headers=auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["file_type"] == "application/pdf"
        assert "id" in data
        assert "user_id" in data
        assert "file_path" in data
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_list_documents(client: TestClient, auth_headers: dict[str, str]) -> None:
    """
    Test listing documents.

    Args:
        client: Test client.
        auth_headers: Authentication headers.

    """
    # Upload a test document first
    test_file_path = "test.pdf"
    with open(test_file_path, "wb") as f:
        f.write(b"%PDF-1.4\n%Test PDF file")

    try:
        with open(test_file_path, "rb") as f:
            client.post(
                "/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                headers=auth_headers,
            )

        # List documents
        response = client.get("/documents/", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["filename"] == "test.pdf"
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_get_document(client: TestClient, auth_headers: dict[str, str]) -> None:
    """
    Test getting a specific document.

    Args:
        client: Test client.
        auth_headers: Authentication headers.

    """
    # Upload a test document first
    test_file_path = "test.pdf"
    with open(test_file_path, "wb") as f:
        f.write(b"%PDF-1.4\n%Test PDF file")

    try:
        with open(test_file_path, "rb") as f:
            upload_response = client.post(
                "/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                headers=auth_headers,
            )

        document_id = upload_response.json()["id"]

        # Get document
        response = client.get(f"/documents/{document_id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == document_id
        assert data["filename"] == "test.pdf"
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


def test_upload_invalid_file_type(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """
    Test uploading an invalid file type.

    Args:
        client: Test client.
        auth_headers: Authentication headers.

    """
    # Create a test file with invalid type
    test_file_path = "test.txt"
    with open(test_file_path, "wb") as f:
        f.write(b"Test file")

    try:
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
                headers=auth_headers,
            )

        assert response.status_code == 400
        assert "File type not supported" in response.json()["detail"]
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
