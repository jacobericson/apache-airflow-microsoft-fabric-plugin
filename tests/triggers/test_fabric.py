from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from airflow.exceptions import AirflowException
from apache_airflow_microsoft_fabric_plugin.hooks.fabric import FabricRunItemStatus
from apache_airflow_microsoft_fabric_plugin.triggers.fabric import FabricTrigger

MODULE = "apache_airflow_microsoft_fabric_plugin.triggers.fabric"


def test_trigger_serialization_includes_retry_params():
    """Test that FabricTrigger.serialize() includes max_api_retries and api_retry_delay."""
    trigger = FabricTrigger(
        fabric_conn_id="fabric_default",
        item_run_id="run1",
        workspace_id="ws",
        item_id="item",
        job_type="RunNotebook",
        end_time=time.time() + 100,
        max_api_retries=7,
        api_retry_delay=2,
    )
    _, kwargs = trigger.serialize()
    assert kwargs["max_api_retries"] == 7
    assert kwargs["api_retry_delay"] == 2


def test_trigger_default_retry_params():
    """Test that FabricTrigger has correct default retry params."""
    trigger = FabricTrigger(
        fabric_conn_id="fabric_default",
        item_run_id="run1",
        workspace_id="ws",
        item_id="item",
        job_type="RunNotebook",
        end_time=time.time() + 100,
    )
    assert trigger.max_api_retries == 5
    assert trigger.api_retry_delay == 1



# =============================================================================
# Trigger run() Tests
# =============================================================================

class TestTriggerRun:
    """Tests for FabricTrigger.run() async generator."""

    def _make_trigger(self, end_time_offset=100, **kwargs):
        defaults = dict(
            fabric_conn_id="fabric_default",
            item_run_id="run-123",
            workspace_id="ws",
            item_id="item",
            job_type="RunNotebook",
            end_time=time.time() + end_time_offset,
            check_interval=1,
            max_api_retries=3,
            api_retry_delay=1,
        )
        defaults.update(kwargs)
        return FabricTrigger(**defaults)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_success_on_first_poll(self, mock_hook_class):
        """Test that trigger yields success when item completes on first poll."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(return_value={
            "status": FabricRunItemStatus.COMPLETED,
        })
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "success"
        assert events[0].payload["item_run_status"] == "Completed"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_success_after_in_progress(self, mock_hook_class, mock_sleep):
        """Test that trigger polls until item completes."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(side_effect=[
            {"status": FabricRunItemStatus.IN_PROGRESS},
            {"status": FabricRunItemStatus.COMPLETED},
        ])
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "success"
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_failure_state(self, mock_hook_class):
        """Test that trigger yields error when item fails."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(return_value={
            "status": FabricRunItemStatus.FAILED,
        })
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert events[0].payload["item_run_status"] == "Failed"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_timeout(self, mock_hook_class):
        """Test that trigger yields timeout error when end_time is reached."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(return_value={
            "status": FabricRunItemStatus.IN_PROGRESS,
        })
        mock_hook_class.return_value = mock_hook

        # Set end_time to the past so timeout is immediate
        trigger = self._make_trigger(end_time_offset=-1)
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert "Timeout" in events[0].payload["message"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_exception_triggers_cancel(self, mock_hook_class):
        """Test that unexpected exception triggers item cancellation."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(
            side_effect=AirflowException("Connection lost")
        )
        mock_hook.cancel_item_run = AsyncMock()
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert events[0].payload["item_run_status"] == FabricRunItemStatus.CANCELLED
        mock_hook.cancel_item_run.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_exception_during_cancel(self, mock_hook_class):
        """Test that exception during cancellation still yields an error event."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(
            side_effect=AirflowException("Connection lost")
        )
        mock_hook.cancel_item_run = AsyncMock(
            side_effect=AirflowException("Cancel also failed")
        )
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert events[0].payload["item_run_status"] == "unknown"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_passes_retry_params_to_hook(self, mock_hook_class):
        """Test that trigger passes max_api_retries and api_retry_delay to FabricAsyncHook."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(return_value={
            "status": FabricRunItemStatus.COMPLETED,
        })
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger(max_api_retries=9, api_retry_delay=3)
        async for _ in trigger.run():
            pass

        mock_hook_class.assert_called_once_with(
            fabric_conn_id="fabric_default",
            max_api_retries=9,
            api_retry_delay=3,
        )

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_cancelled_state(self, mock_hook_class):
        """Test that trigger yields error when item is cancelled."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(return_value={
            "status": FabricRunItemStatus.CANCELLED,
        })
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert events[0].payload["item_run_status"] == "Cancelled"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_deduped_state(self, mock_hook_class):
        """Test that trigger yields error when item is deduped."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(return_value={
            "status": FabricRunItemStatus.DEDUPED,
        })
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        events = []
        async for event in trigger.run():
            events.append(event)

        assert len(events) == 1
        assert events[0].payload["status"] == "error"
        assert events[0].payload["item_run_status"] == "Deduped"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.FabricAsyncHook")
    async def test_run_programming_error_propagates(self, mock_hook_class):
        """Test that non-AirflowException errors propagate instead of being caught."""
        mock_hook = MagicMock()
        mock_hook.async_get_item_run_details = AsyncMock(
            side_effect=KeyError("missing key")
        )
        mock_hook_class.return_value = mock_hook

        trigger = self._make_trigger()
        with pytest.raises(KeyError, match="missing key"):
            async for _ in trigger.run():
                pass
