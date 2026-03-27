from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from airflow.exceptions import AirflowException, TaskDeferred
from apache_airflow_microsoft_fabric_plugin.hooks.fabric import FabricRunItemStatus
from apache_airflow_microsoft_fabric_plugin.operators.fabric import FabricRunItemOperator
from apache_airflow_microsoft_fabric_plugin.triggers.fabric import FabricTrigger


def test_operator_stores_retry_params():
    """Test that FabricRunItemOperator stores max_api_retries and api_retry_delay."""
    op = FabricRunItemOperator(
        task_id="test",
        workspace_id="ws",
        item_id="item",
        fabric_conn_id="fabric_default",
        job_type="RunNotebook",
        max_api_retries=10,
        api_retry_delay=3,
    )
    assert op.max_api_retries == 10
    assert op.api_retry_delay == 3


def test_operator_default_retry_params():
    """Test that FabricRunItemOperator has correct default retry params."""
    op = FabricRunItemOperator(
        task_id="test",
        workspace_id="ws",
        item_id="item",
        fabric_conn_id="fabric_default",
        job_type="RunNotebook",
    )
    assert op.max_api_retries == 5
    assert op.api_retry_delay == 1


def test_operator_hook_receives_retry_params():
    """Test that the hook cached property receives retry params from operator."""
    op = FabricRunItemOperator(
        task_id="test",
        workspace_id="ws",
        item_id="item",
        fabric_conn_id="fabric_default",
        job_type="RunNotebook",
        max_api_retries=8,
        api_retry_delay=4,
    )
    hook = op.hook
    assert hook.max_api_retries == 8
    assert hook.api_retry_delay == 4


def test_operator_defers_with_retry_params():
    """Test that deferrable mode passes retry params to the trigger."""
    op = FabricRunItemOperator(
        task_id="test",
        workspace_id="ws",
        item_id="item",
        fabric_conn_id="fabric_default",
        job_type="RunNotebook",
        max_api_retries=7,
        api_retry_delay=2,
        deferrable=True,
    )

    # Mock the hook methods
    mock_hook = MagicMock()
    mock_hook.run_fabric_item.return_value = "https://api.fabric.microsoft.com/location"
    mock_hook.get_item_run_details.return_value = {
        "status": FabricRunItemStatus.IN_PROGRESS,
        "id": "run-123",
    }

    # Mock context
    mock_ti = MagicMock()
    mock_context = {"ti": mock_ti}

    with patch.object(type(op), "hook", new_callable=lambda: property(lambda self: mock_hook)):
        with pytest.raises(TaskDeferred) as exc_info:
            op.execute(mock_context)

        trigger = exc_info.value.trigger
        assert isinstance(trigger, FabricTrigger)
        assert trigger.max_api_retries == 7
        assert trigger.api_retry_delay == 2


def test_execute_complete_success():
    """Test that execute_complete handles success events correctly."""
    op = FabricRunItemOperator(
        task_id="test",
        workspace_id="ws",
        item_id="item",
        fabric_conn_id="fabric_default",
        job_type="RunNotebook",
    )
    mock_ti = MagicMock()
    mock_context = {"ti": mock_ti}
    event = {
        "status": "success",
        "message": "The item run run-123 has status Completed.",
        "run_id": "run-123",
        "item_run_status": "Completed",
    }

    op.execute_complete(mock_context, event)

    mock_ti.xcom_push.assert_called_once_with(key="run_status", value="Completed")


def test_execute_complete_error():
    """Test that execute_complete raises FabricRunItemException on error events."""
    from apache_airflow_microsoft_fabric_plugin.hooks.fabric import FabricRunItemException

    op = FabricRunItemOperator(
        task_id="test",
        workspace_id="ws",
        item_id="item",
        fabric_conn_id="fabric_default",
        job_type="RunNotebook",
    )
    mock_ti = MagicMock()
    mock_context = {"ti": mock_ti}
    event = {
        "status": "error",
        "message": "The item run run-123 has status Failed.",
        "run_id": "run-123",
        "item_run_status": "Failed",
    }

    with pytest.raises(FabricRunItemException, match="Failed"):
        op.execute_complete(mock_context, event)


