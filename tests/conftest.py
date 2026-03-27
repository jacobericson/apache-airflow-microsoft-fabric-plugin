from __future__ import annotations

from unittest.mock import patch

import pytest
from airflow.models.connection import Connection


@pytest.fixture
def create_mock_connection():
    """Fixture to mock Airflow connection retrieval."""
    connections = {}

    def _create_mock_connection(connection):
        connections[connection.conn_id] = connection

    with patch("airflow.hooks.base.BaseHook.get_connection") as mock_get_connection:
        mock_get_connection.side_effect = lambda conn_id: connections.get(conn_id)
        yield _create_mock_connection


@pytest.fixture(autouse=True)
def setup_connections(create_mock_connection):
    """Shared fixture that registers the default Fabric connection for all tests."""
    create_mock_connection(
        Connection(
            conn_id="fabric_default",
            conn_type="generic",
            login="clientId",
            password="userRefreshToken",
            extra={
                "tenantId": "tenantId",
            },
        )
    )
