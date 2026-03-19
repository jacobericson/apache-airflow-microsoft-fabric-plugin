from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, patch, AsyncMock

import aiohttp
import pytest
import requests

from airflow.exceptions import AirflowException
from airflow.models.connection import Connection
from apache_airflow_microsoft_fabric_plugin.hooks.fabric import (
    DEFAULT_RETRY_AFTER_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    FabricAsyncHook,
    FabricHook,
    FabricRunItemException,
    FabricRunItemStatus,
    RETRYABLE_STATUS_CODES,
    _calculate_backoff,
    _get_retry_after_seconds,
    _should_retry_connection_error,
    _should_retry_status,
)

DEFAULT_FABRIC_CONNECTION = "fabric_default"
ITEM_RUN_LOCATION = "https://api.fabric.microsoft.com/v1/workspaces/4b218778-e7a5-4d73-8187-f10824047715/items/431e8d7b-4a95-4c02-8ccd-6faef5ba1bd7/jobs/instances/f2d65699-dd22-4889-980c-15226deb0e1b"
WORKSPACE_ID = "workspace_id"
ITEM_ID = "item_id"
ITEM_RUN_ID = "item_run_id"
BASE_URL = "https://api.fabric.microsoft.com"
API_VERSION = "v1"
JOB_TYPE = "RunNotebook"
MODULE = "apache_airflow_microsoft_fabric_plugin.hooks.fabric"


@pytest.fixture
def fabric_hook():
    client = FabricHook(fabric_conn_id=DEFAULT_FABRIC_CONNECTION)
    return client


@pytest.fixture
def get_token(fabric_hook):
    fabric_hook._get_token = MagicMock(return_value="access_token")
    return fabric_hook._get_token


def test_get_headers(get_token, fabric_hook):
    headers = fabric_hook.get_headers()
    assert isinstance(headers, dict)
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer access_token"


def test_get_item_run_details_success(fabric_hook, get_token, mocker):
    # Mock response for successful response from _send_request
    response = MagicMock()
    response.ok = True
    response.json.return_value = {"status": "Completed"}

    mocker.patch.object(fabric_hook, "_send_request", return_value=response)
    mocker.patch.object(fabric_hook, "get_headers", return_value={"Authorization": "Bearer access_token"})

    result = fabric_hook.get_item_run_details(location=ITEM_RUN_LOCATION)

    assert result == {"status": "Completed"}
    fabric_hook.get_headers.assert_called_once()
    fabric_hook._send_request.assert_called_once_with(
        "GET", ITEM_RUN_LOCATION, headers={"Authorization": f"Bearer {get_token.return_value}"}
    )


def test_get_item_run_details_failure(create_mock_connection, mocker):
    # Create a hook with max_api_retries=1 to avoid long retry delays in test
    create_mock_connection(
        Connection(
            conn_id="test_failure_conn",
            conn_type="generic",
            login="clientId",
            password="userRefreshToken",
            extra={"tenantId": "tenantId"},
        )
    )
    hook = FabricHook(fabric_conn_id="test_failure_conn", max_api_retries=1)

    # _send_request now raises AirflowException for non-retryable errors
    mocker.patch.object(
        hook, "_send_request",
        side_effect=AirflowException("Request failed with status 400"),
    )
    mocker.patch.object(hook, "_get_token", return_value="access_token")

    with pytest.raises(AirflowException):
        hook.get_item_run_details(location=ITEM_RUN_LOCATION)


@patch(f"{MODULE}.FabricHook._send_request")
def test_get_item_details(mock_send_request, fabric_hook, get_token):
    fabric_hook.get_item_details(WORKSPACE_ID, ITEM_ID)
    expected_url = f"{BASE_URL}/{API_VERSION}/workspaces/{WORKSPACE_ID}/items/{ITEM_ID}"
    mock_send_request.assert_called_once_with(
        "GET", expected_url, headers={"Authorization": f"Bearer {get_token.return_value}"}
    )


@patch(f"{MODULE}.FabricHook._send_request")
def test_run_fabric_item(mock_send_request, fabric_hook, get_token):
    mock_response = MagicMock()
    mock_response.headers = {"Location": "https://api.fabric.microsoft.com/location"}
    mock_send_request.return_value = mock_response

    result = fabric_hook.run_fabric_item(WORKSPACE_ID, ITEM_ID, JOB_TYPE, job_params=None)
    expected_url = f"{BASE_URL}/{API_VERSION}/workspaces/{WORKSPACE_ID}/items/{ITEM_ID}/jobs/instances?jobType={JOB_TYPE}"
    mock_send_request.assert_called_once_with(
        "POST", expected_url, headers={"Authorization": f"Bearer {get_token.return_value}"}, json={}
    )
    assert result == "https://api.fabric.microsoft.com/location"

_wait_for_item_run_status_test_args = [
    (FabricRunItemStatus.COMPLETED, FabricRunItemStatus.COMPLETED, True),
    (FabricRunItemStatus.FAILED, FabricRunItemStatus.COMPLETED, False),
    (FabricRunItemStatus.IN_PROGRESS, FabricRunItemStatus.COMPLETED, "timeout"),
    (FabricRunItemStatus.NOT_STARTED, FabricRunItemStatus.COMPLETED, "timeout"),
    (FabricRunItemStatus.CANCELLED, FabricRunItemStatus.COMPLETED, False),
    # Deduped is in FAILURE_STATES but NOT in TERMINAL_STATUSES, so sync path loops until timeout
    (FabricRunItemStatus.DEDUPED, FabricRunItemStatus.COMPLETED, "timeout"),
]

@pytest.mark.parametrize(
    argnames=("item_run_status", "expected_status", "expected_result"),
    argvalues=_wait_for_item_run_status_test_args,
    ids=[
        f"run_status_{argval[0]}_expected_{argval[1]}"
        if isinstance(argval[1], str)
        else f"run_status_{argval[0]}_expected_AnyTerminalStatus"
        for argval in _wait_for_item_run_status_test_args
    ]
)
def test_wait_for_item_run_status(fabric_hook, item_run_status, expected_status, expected_result):
    config = {
        "location": ITEM_RUN_LOCATION,
        "timeout": 3,
        "check_interval": 1,
        "target_status": expected_status,
    }

    with patch.object(FabricHook, "get_item_run_details") as mock_item_run:
        mock_item_run.return_value = {"status": item_run_status}

        if expected_result != "timeout":
            assert fabric_hook.wait_for_item_run_status(**config) == expected_result
        else:
            with (
                patch(f"{MODULE}.time.monotonic", side_effect=[0, 0, 1, 1, 2, 2, 3, 3, 4]),
                patch(f"{MODULE}.time.sleep"),
            ):
                with pytest.raises(FabricRunItemException):
                    fabric_hook.wait_for_item_run_status(**config)

@pytest.fixture
def fabric_async_hook():
    client = FabricAsyncHook(fabric_conn_id=DEFAULT_FABRIC_CONNECTION)
    return client

@pytest.mark.asyncio
async def test_async_get_headers(fabric_async_hook, mocker):
    mocker.patch.object(fabric_async_hook, "_async_get_token", return_value="access_token")
    headers = await fabric_async_hook.async_get_headers()
    assert isinstance(headers, dict)
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer access_token"


@pytest.mark.asyncio
async def test_async_get_item_run_details_success(fabric_async_hook, mocker):
    # Mock async methods properly
    mocker.patch.object(fabric_async_hook, "async_get_headers", return_value={"Authorization": "Bearer access_token"})
    mocker.patch.object(fabric_async_hook, "_async_send_request", return_value={"status": "Completed"})

    expected_url = f"{BASE_URL}/{API_VERSION}/workspaces/{WORKSPACE_ID}/items/{ITEM_ID}/jobs/instances/{ITEM_RUN_ID}"
    result = await fabric_async_hook.async_get_item_run_details(workspace_id=WORKSPACE_ID, item_id=ITEM_ID, item_run_id=ITEM_RUN_ID)

    assert result == {"status": "Completed"}
    fabric_async_hook.async_get_headers.assert_called_once()
    fabric_async_hook._async_send_request.assert_called_once_with(
        "GET", expected_url, headers={"Authorization": "Bearer access_token"}
    )

@pytest.mark.asyncio
async def test_async_cancel_item_run(fabric_async_hook, mocker):
    mocker.patch.object(fabric_async_hook, "async_get_headers", return_value={"Authorization": "Bearer access_token"})
    mock_response = MagicMock()
    mocker.patch.object(fabric_async_hook, "_async_send_request", return_value=mock_response)

    await fabric_async_hook.cancel_item_run(WORKSPACE_ID, ITEM_ID, ITEM_RUN_ID)
    expected_url = f"{BASE_URL}/{API_VERSION}/workspaces/{WORKSPACE_ID}/items/{ITEM_ID}/jobs/instances/{ITEM_RUN_ID}/cancel"
    fabric_async_hook._async_send_request.assert_called_once_with(
        "POST", expected_url, headers={"Authorization": "Bearer access_token"}, idempotent=True
    )


# =============================================================================
# Retry Behavior Tests for Sync Hook
# =============================================================================

class TestSendRequestRetry:
    """Tests for _send_request retry behavior on transient errors."""

    @pytest.fixture
    def hook_with_retries(self, create_mock_connection):
        """Create a hook with explicit retry settings for testing."""
        create_mock_connection(
            Connection(
                conn_id="retry_test_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricHook(fabric_conn_id="retry_test_conn", max_api_retries=3, api_retry_delay=1)

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_retries_on_429_rate_limit(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that 429 status code triggers retry and eventually succeeds."""
        # First two calls return 429, third succeeds
        rate_limit_response = MagicMock()
        rate_limit_response.ok = False
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "2"}

        success_response = MagicMock()
        success_response.ok = True
        success_response.status_code = 200

        mock_get.side_effect = [rate_limit_response, rate_limit_response, success_response]

        result = hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert result == success_response
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
        # Retry-After=2 is the floor; attempt 1 backoff=1, attempt 2 backoff=2. max(backoff, 2) = 2 both times
        mock_sleep.assert_any_call(2.0)

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_retries_on_503_server_error(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that 503 status code triggers retry."""
        server_error_response = MagicMock()
        server_error_response.ok = False
        server_error_response.status_code = 503
        server_error_response.headers = {}

        success_response = MagicMock()
        success_response.ok = True

        mock_get.side_effect = [server_error_response, success_response]

        result = hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert result == success_response
        assert mock_get.call_count == 2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that connection errors trigger retry."""
        success_response = MagicMock()
        success_response.ok = True

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            success_response,
        ]

        result = hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert result == success_response
        assert mock_get.call_count == 2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_raises_after_max_api_retries_exhausted(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that after max retries, AirflowException is raised."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(AirflowException, match="failed"):
            hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert mock_get.call_count == 3  # max_api_retries = 3
        assert mock_sleep.call_count == 2  # No sleep after final attempt

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.requests.get")
    def test_no_retry_on_400_bad_request(self, mock_get, mock_random, hook_with_retries):
        """Test that 400 errors are not retried and raise AirflowException."""
        bad_request_response = MagicMock()
        bad_request_response.ok = False
        bad_request_response.status_code = 400
        bad_request_response.text = "Bad Request"
        bad_request_response.headers = {}

        mock_get.return_value = bad_request_response

        with pytest.raises(AirflowException, match="400"):
            hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert mock_get.call_count == 1  # No retries

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_uses_exponential_backoff_when_retry_after_header_missing(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that exponential backoff is used when Retry-After header is missing."""
        rate_limit_response = MagicMock()
        rate_limit_response.ok = False
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}  # No Retry-After header

        success_response = MagicMock()
        success_response.ok = True

        mock_get.side_effect = [rate_limit_response, success_response]

        hook_with_retries._send_request("GET", "https://api.example.com/test")

        # No Retry-After header -> retry_after=None -> pure exponential: 1 * 2^0 = 1.0
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.parametrize("status_code", list(RETRYABLE_STATUS_CODES))
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_all_retryable_status_codes(self, mock_get, mock_sleep, mock_random, status_code, hook_with_retries):
        """Test that all defined retryable status codes trigger retry."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = status_code
        error_response.headers = {}

        success_response = MagicMock()
        success_response.ok = True

        mock_get.side_effect = [error_response, success_response]

        result = hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert result == success_response
        assert mock_get.call_count == 2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_retries_on_timeout(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that Timeout exceptions trigger retry."""
        success_response = MagicMock()
        success_response.ok = True

        mock_get.side_effect = [
            requests.exceptions.Timeout("Read timed out"),
            success_response,
        ]

        result = hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert result == success_response
        assert mock_get.call_count == 2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_raises_after_timeout_retries_exhausted(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that Timeout raises AirflowException after max retries."""
        mock_get.side_effect = requests.exceptions.Timeout("Read timed out")

        with pytest.raises(AirflowException, match="failed"):
            hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # No sleep after final attempt

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_raises_after_retryable_status_exhausted(self, mock_get, mock_sleep, mock_random, hook_with_retries):
        """Test that retryable status codes raise AirflowException after max retries."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 503
        error_response.headers = {}

        mock_get.return_value = error_response

        with pytest.raises(AirflowException, match="503"):
            hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2  # No sleep after final attempt

    @patch(f"{MODULE}.requests.get")
    def test_send_request_success(self, mock_get, hook_with_retries):
        """Test _send_request returns response on success."""
        success_response = MagicMock()
        success_response.ok = True
        mock_get.return_value = success_response

        result = hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert result == success_response
        mock_get.assert_called_once_with(url="https://api.example.com/test")

    @patch(f"{MODULE}.requests.get")
    def test_send_request_passes_kwargs(self, mock_get, hook_with_retries):
        """Test _send_request forwards kwargs to underlying request function."""
        success_response = MagicMock()
        success_response.ok = True
        mock_get.return_value = success_response

        headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
        hook_with_retries._send_request("GET", "https://api.example.com/test", headers=headers)

        mock_get.assert_called_once_with(
            url="https://api.example.com/test",
            headers=headers,
        )

    def test_unsupported_request_type_raises(self, hook_with_retries):
        """Test that unsupported request types raise AirflowException."""
        with pytest.raises(AirflowException, match="Unsupported request type"):
            hook_with_retries._send_request("DELETE", "https://api.example.com/test")

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.requests.post")
    def test_post_no_retry_on_503(self, mock_post, mock_random, hook_with_retries):
        """Test that POST requests do NOT retry on 503 server errors."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 503
        error_response.text = "Service Unavailable"
        error_response.headers = {}

        mock_post.return_value = error_response

        with pytest.raises(AirflowException, match="503"):
            hook_with_retries._send_request("POST", "https://api.example.com/test", json={"key": "val"})

        assert mock_post.call_count == 1  # No retries for POST on 5xx

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.requests.post")
    def test_post_no_retry_on_connection_error(self, mock_post, mock_random, hook_with_retries):
        """Test that POST requests do NOT retry on connection errors."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(AirflowException, match="WARNING: The server may have already processed"):
            hook_with_retries._send_request("POST", "https://api.example.com/test")

        assert mock_post.call_count == 1  # No retries for POST on connection error

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.requests.post")
    def test_post_no_retry_on_timeout(self, mock_post, mock_random, hook_with_retries):
        """Test that POST requests do NOT retry on timeout errors."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out")

        with pytest.raises(AirflowException, match="WARNING: The server may have already processed"):
            hook_with_retries._send_request("POST", "https://api.example.com/test")

        assert mock_post.call_count == 1  # No retries for POST on timeout

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.post")
    def test_post_retries_on_429(self, mock_post, mock_sleep, mock_random, hook_with_retries):
        """Test that POST requests DO retry on 429 rate limit."""
        rate_limit_response = MagicMock()
        rate_limit_response.ok = False
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"Retry-After": "1"}

        success_response = MagicMock()
        success_response.ok = True

        mock_post.side_effect = [rate_limit_response, success_response]

        result = hook_with_retries._send_request("POST", "https://api.example.com/test")

        assert result == success_response
        assert mock_post.call_count == 2  # Retried on 429

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.requests.get")
    def test_non_retryable_error_truncates_response_text(self, mock_get, mock_random, hook_with_retries):
        """Test that non-retryable error messages truncate long response text."""
        long_text = "X" * 1000
        bad_response = MagicMock()
        bad_response.ok = False
        bad_response.status_code = 400
        bad_response.text = long_text
        bad_response.headers = {}

        mock_get.return_value = bad_response

        with pytest.raises(AirflowException) as exc_info:
            hook_with_retries._send_request("GET", "https://api.example.com/test")

        assert "X" * 500 in str(exc_info.value)
        assert "X" * 501 not in str(exc_info.value)

    # --- Idempotent POST tests (Issue #1) ---

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.post")
    def test_post_retries_on_503_when_idempotent(self, mock_post, mock_sleep, mock_random, hook_with_retries):
        """Test that POST + idempotent=True retries on 503."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 503
        error_response.headers = {}

        success_response = MagicMock()
        success_response.ok = True

        mock_post.side_effect = [error_response, success_response]

        result = hook_with_retries._send_request("POST", "https://api.example.com/test", idempotent=True)

        assert result == success_response
        assert mock_post.call_count == 2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.post")
    def test_post_retries_on_connection_error_when_idempotent(self, mock_post, mock_sleep, mock_random, hook_with_retries):
        """Test that POST + idempotent=True retries on ConnectionError."""
        success_response = MagicMock()
        success_response.ok = True

        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            success_response,
        ]

        result = hook_with_retries._send_request("POST", "https://api.example.com/test", idempotent=True)

        assert result == success_response
        assert mock_post.call_count == 2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.post")
    def test_post_retries_on_timeout_when_idempotent(self, mock_post, mock_sleep, mock_random, hook_with_retries):
        """Test that POST + idempotent=True retries on Timeout."""
        success_response = MagicMock()
        success_response.ok = True

        mock_post.side_effect = [
            requests.exceptions.Timeout("Read timed out"),
            success_response,
        ]

        result = hook_with_retries._send_request("POST", "https://api.example.com/test", idempotent=True)

        assert result == success_response
        assert mock_post.call_count == 2


# =============================================================================
# Retry Behavior Tests for Async Hook
# =============================================================================

class TestAsyncSendRequestRetry:
    """Tests for _async_send_request retry behavior on transient errors."""

    @pytest.fixture
    def async_hook_with_retries(self, create_mock_connection):
        """Create an async hook with explicit retry settings for testing."""
        create_mock_connection(
            Connection(
                conn_id="async_retry_test_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricAsyncHook(fabric_conn_id="async_retry_test_conn", max_api_retries=3, api_retry_delay=1)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_retries_on_429_rate_limit(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that async hook retries on 429 and eventually succeeds."""
        rate_limit_response = MagicMock()
        rate_limit_response.status = 429
        rate_limit_response.headers = {"Retry-After": "2", "Content-Type": "application/json"}

        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[rate_limit_response, rate_limit_response, success_response])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert result == {"status": "success"}
        assert mock_session.get.call_count == 3
        assert mock_sleep.call_count == 2
        # Retry-After=2 is the floor; attempt 1 backoff=1, attempt 2 backoff=2. max(backoff, 2) = 2 both times
        mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_retries_on_connection_error(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that async hook retries on connection errors."""
        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[
            aiohttp.ClientConnectionError("Connection error"),
            success_response,
        ])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert result == {"status": "success"}
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_raises_after_max_api_retries(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that async hook raises AirflowException after max retries exhausted."""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=aiohttp.ClientConnectionError("Connection error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="failed"):
            await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert mock_session.get.call_count == 3  # max_api_retries = 3
        assert mock_sleep.call_count == 2  # No sleep after final attempt

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_retries_on_timeout_error(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that asyncio.TimeoutError triggers retry."""
        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[
            asyncio.TimeoutError(),
            success_response,
        ])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert result == {"status": "success"}
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_raises_after_retryable_status_exhaustion(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that async hook raises AirflowException after retryable status exhaustion."""
        error_response = MagicMock()
        error_response.status = 503
        error_response.headers = {}

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=error_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="503"):
            await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert mock_session.get.call_count == 3
        assert mock_sleep.call_count == 2  # No sleep after final attempt

    @pytest.mark.asyncio
    async def test_async_unsupported_request_type_raises(self, async_hook_with_retries):
        """Test that unsupported request types raise AirflowException in async."""
        with pytest.raises(AirflowException, match="Unsupported request type"):
            await async_hook_with_retries._async_send_request("DELETE", "https://api.example.com/test")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_no_retry_on_non_retryable_status(self, mock_session_class, mock_random, async_hook_with_retries):
        """Test that non-retryable 4xx errors raise immediately without retrying."""
        error_response = MagicMock()
        error_response.status = 401
        error_response.headers = {}
        error_response.text = AsyncMock(return_value="Unauthorized")

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=error_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="401"):
            await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert mock_session.get.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", list(RETRYABLE_STATUS_CODES))
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_all_retryable_status_codes(self, mock_session_class, mock_sleep, mock_random, status_code, async_hook_with_retries):
        """Test that all defined retryable status codes trigger retry in async."""
        error_response = MagicMock()
        error_response.status = status_code
        error_response.headers = {}

        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=[error_response, success_response])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert result == {"status": "success"}
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_no_retry_on_server_error(self, mock_session_class, mock_random, async_hook_with_retries):
        """Test that POST requests do NOT retry on 503 server errors in async."""
        error_response = MagicMock()
        error_response.status = 503
        error_response.headers = {}
        error_response.text = AsyncMock(return_value="Service Unavailable")

        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=error_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="503"):
            await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test")

        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_no_retry_on_connection_error(self, mock_session_class, mock_random, async_hook_with_retries):
        """Test that POST requests do NOT retry on connection errors in async."""
        mock_session = MagicMock()
        mock_session.post = AsyncMock(side_effect=aiohttp.ClientConnectionError("Connection error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="WARNING: The server may have already processed"):
            await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test")

        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_no_retry_on_timeout(self, mock_session_class, mock_random, async_hook_with_retries):
        """Test that POST requests do NOT retry on timeout errors in async."""
        mock_session = MagicMock()
        mock_session.post = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="WARNING: The server may have already processed"):
            await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test")

        assert mock_session.post.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_retries_on_429(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that POST requests DO retry on 429 rate limit in async."""
        rate_limit_response = MagicMock()
        rate_limit_response.status = 429
        rate_limit_response.headers = {"Retry-After": "1"}

        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.post = AsyncMock(side_effect=[rate_limit_response, success_response])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test")

        assert result == {"status": "success"}
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_unsupported_content_type_raises(self, mock_session_class, async_hook_with_retries):
        """Test that unsupported Content-Type raises AirflowException."""
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "text/html"}

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="Unsupported Content-Type"):
            await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_octet_stream_returns_bytes(self, mock_session_class, async_hook_with_retries):
        """Test that application/octet-stream returns raw bytes."""
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "application/octet-stream"}
        response.read = AsyncMock(return_value=b"binary data")

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert result == b"binary data"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_success_json_response(self, mock_session_class, async_hook_with_retries):
        """Test that successful JSON responses are parsed correctly."""
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "application/json"}
        response.json = AsyncMock(return_value={"key": "value"})

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_non_retryable_error_truncates_response_text(self, mock_session_class, mock_random, async_hook_with_retries):
        """Test that non-retryable error messages truncate long response text."""
        long_text = "X" * 1000
        error_response = MagicMock()
        error_response.status = 400
        error_response.headers = {}
        error_response.text = AsyncMock(return_value=long_text)

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=error_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException) as exc_info:
            await async_hook_with_retries._async_send_request("GET", "https://api.example.com/test")

        assert "X" * 500 in str(exc_info.value)
        assert "X" * 501 not in str(exc_info.value)

    # --- Idempotent POST tests (Issue #1) ---

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_retries_on_503_when_idempotent(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that POST + idempotent=True retries on 503 in async."""
        error_response = MagicMock()
        error_response.status = 503
        error_response.headers = {}

        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.post = AsyncMock(side_effect=[error_response, success_response])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test", idempotent=True)

        assert result == {"status": "success"}
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_retries_on_connection_error_when_idempotent(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that POST + idempotent=True retries on connection error in async."""
        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.post = AsyncMock(side_effect=[
            aiohttp.ClientConnectionError("Connection error"),
            success_response,
        ])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test", idempotent=True)

        assert result == {"status": "success"}
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_post_retries_on_timeout_when_idempotent(self, mock_session_class, mock_sleep, mock_random, async_hook_with_retries):
        """Test that POST + idempotent=True retries on timeout in async."""
        success_response = MagicMock()
        success_response.status = 200
        success_response.headers = {"Content-Type": "application/json"}
        success_response.json = AsyncMock(return_value={"status": "success"})

        mock_session = MagicMock()
        mock_session.post = AsyncMock(side_effect=[
            asyncio.TimeoutError(),
            success_response,
        ])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook_with_retries._async_send_request("POST", "https://api.example.com/test", idempotent=True)

        assert result == {"status": "success"}
        assert mock_session.post.call_count == 2


# =============================================================================
# _get_retry_after_seconds Tests
# =============================================================================

class TestGetRetryAfterSeconds:
    """Tests for the _get_retry_after_seconds helper function."""

    def test_valid_numeric_header(self):
        assert _get_retry_after_seconds({"Retry-After": "10"}) == 10.0

    def test_valid_float_header(self):
        assert _get_retry_after_seconds({"Retry-After": "2.5"}) == 2.5

    def test_missing_header(self):
        assert _get_retry_after_seconds({}) is None

    def test_invalid_non_numeric_header(self):
        assert _get_retry_after_seconds({"Retry-After": "not-a-number"}) == DEFAULT_RETRY_AFTER_SECONDS

    def test_empty_header(self):
        assert _get_retry_after_seconds({"Retry-After": ""}) is None

    def test_http_date_header(self):
        """Test that HTTP-date format Retry-After is parsed correctly."""
        frozen_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        future = frozen_now + timedelta(seconds=30)
        date_str = format_datetime(future)

        with patch(f"{MODULE}.datetime") as mock_dt:
            mock_dt.now.return_value = frozen_now
            # Allow parsedate_to_datetime to still work with real datetime
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _get_retry_after_seconds({"Retry-After": date_str})

        assert result == pytest.approx(30.0, abs=1.0)

    def test_http_date_in_past(self):
        """Test that past HTTP-date returns 0."""
        frozen_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        past = frozen_now - timedelta(seconds=60)
        date_str = format_datetime(past)

        with patch(f"{MODULE}.datetime") as mock_dt:
            mock_dt.now.return_value = frozen_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _get_retry_after_seconds({"Retry-After": date_str})

        assert result == 0

    def test_invalid_header_logs_warning(self):
        """Test that unparseable Retry-After logs a warning."""
        with patch(f"{MODULE}.logger") as mock_logger:
            result = _get_retry_after_seconds({"Retry-After": "not-a-number"})
            assert result == DEFAULT_RETRY_AFTER_SECONDS
            mock_logger.warning.assert_called_once()

    def test_negative_numeric_header_clamped_to_zero(self):
        """Test that negative Retry-After numeric values are clamped to 0."""
        assert _get_retry_after_seconds({"Retry-After": "-5"}) == 0

    def test_numeric_retry_after_capped_at_max(self):
        """Test that Retry-After values exceeding MAX_RETRY_AFTER_SECONDS are clamped."""
        with patch(f"{MODULE}.logger") as mock_logger:
            result = _get_retry_after_seconds({"Retry-After": "600"})
            assert result == float(MAX_RETRY_AFTER_SECONDS)
            mock_logger.warning.assert_called_once()

    def test_numeric_retry_after_under_cap_not_clamped(self):
        """Test that Retry-After values under the cap are not clamped."""
        result = _get_retry_after_seconds({"Retry-After": "299"})
        assert result == 299.0

    def test_http_date_retry_after_capped(self):
        """Test that HTTP-date Retry-After far in the future is clamped."""
        frozen_now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        far_future = frozen_now + timedelta(seconds=600)
        date_str = format_datetime(far_future)

        with patch(f"{MODULE}.datetime") as mock_dt:
            mock_dt.now.return_value = frozen_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with patch(f"{MODULE}.logger") as mock_logger:
                result = _get_retry_after_seconds({"Retry-After": date_str})
                assert result == float(MAX_RETRY_AFTER_SECONDS)
                mock_logger.warning.assert_called_once()


# =============================================================================
# _calculate_backoff Tests
# =============================================================================

class TestCalculateBackoff:
    """Tests for the _calculate_backoff helper function."""

    @patch(f"{MODULE}.random.uniform", return_value=0)
    def test_basic_exponential(self, mock_random):
        assert _calculate_backoff(1, 1) == 1.0    # 1 * 2^0
        assert _calculate_backoff(1, 2) == 2.0    # 1 * 2^1
        assert _calculate_backoff(1, 3) == 4.0    # 1 * 2^2
        assert _calculate_backoff(2, 3) == 8.0    # 2 * 2^2

    @patch(f"{MODULE}.random.uniform", return_value=0)
    def test_retry_after_as_floor(self, mock_random):
        # retry_after=10 > exponential=1, so use 10
        assert _calculate_backoff(1, 1, retry_after=10) == 10.0
        # retry_after=2 < exponential=4, so use 4
        assert _calculate_backoff(1, 3, retry_after=2) == 4.0

    @patch(f"{MODULE}.random.uniform", return_value=0.9)
    def test_max_delay_cap(self, mock_random):
        # Very high attempt: 1 * 2^19 = 524288, capped at 60
        # At cap, uses uniform(0.75, 1.0) -> returns 0.9 -> 60 * 0.9 = 54.0
        assert _calculate_backoff(1, 20) == 54.0

    def test_jitter_is_applied(self):
        """Test that jitter adds variability."""
        with patch(f"{MODULE}.random.uniform", return_value=0.25):
            # delay=1, jitter=1*0.25=0.25, total=1.25
            assert _calculate_backoff(1, 1) == 1.25

    @patch(f"{MODULE}.random.uniform", return_value=0)
    def test_retry_after_none_uses_exponential(self, mock_random):
        assert _calculate_backoff(1, 1, retry_after=None) == 1.0

    @patch(f"{MODULE}.random.uniform", return_value=0.9)
    def test_jitter_preserved_at_effective_max(self, mock_random):
        """Test that jitter is preserved even when delay reaches the cap."""
        # max_delay=60, exponential for attempt 20 >> 60, so delay hits cap
        # At cap, uses effective_max * uniform(0.75, 1.0) -> 60 * 0.9 = 54.0
        result = _calculate_backoff(1, 20, max_delay=60.0)
        assert result == 54.0

    @patch(f"{MODULE}.random.uniform", return_value=0.9)
    def test_retry_after_above_max_delay_uses_retry_after_as_cap(self, mock_random):
        """Test that when retry_after > max_delay, retry_after is used as effective max."""
        # retry_after=120 > max_delay=60, effective_max=120
        # At cap: max(120 * 0.9, 120) = max(108, 120) = 120
        # Floor prevents jitter from going below retry_after
        result = _calculate_backoff(1, 1, retry_after=120, max_delay=60.0)
        assert result == 120.0

    @patch(f"{MODULE}.random.uniform", return_value=0.9)
    def test_jitter_never_undercuts_retry_after(self, mock_random):
        """Test that jitter never reduces delay below the server-specified Retry-After."""
        # retry_after=120, effective_max=120, at cap
        # max(120 * 0.9, 120) = 120 — floor prevents undershoot
        result = _calculate_backoff(1, 1, retry_after=120, max_delay=60.0)
        assert result == 120.0

    @patch(f"{MODULE}.random.uniform", return_value=0.8)
    def test_jitter_allowed_when_retry_after_below_jitter_range(self, mock_random):
        """Test that jitter works normally when retry_after is below the jitter floor."""
        # retry_after=10, max_delay=60, exponential for attempt 20 >> 60, at cap
        # max(60 * 0.8, 10) = max(48, 10) = 48 — jitter applies above the floor
        result = _calculate_backoff(1, 20, retry_after=10, max_delay=60.0)
        assert result == 48.0


# =============================================================================
# _should_retry_status and _should_retry_connection_error Tests
# =============================================================================

class TestShouldRetryHelpers:
    """Tests for the retry decision helper functions."""

    def test_retryable_get_returns_true(self):
        assert _should_retry_status(503, "GET", False) is True

    def test_non_retryable_status_returns_false(self):
        assert _should_retry_status(400, "GET", False) is False

    def test_post_503_not_idempotent_returns_false(self):
        assert _should_retry_status(503, "POST", False) is False

    def test_post_429_not_idempotent_returns_true(self):
        assert _should_retry_status(429, "POST", False) is True

    def test_post_503_idempotent_returns_true(self):
        assert _should_retry_status(503, "POST", True) is True

    def test_get_connection_error_retries(self):
        assert _should_retry_connection_error("GET", False) is True

    def test_post_connection_error_no_retry(self):
        assert _should_retry_connection_error("POST", False) is False

    def test_post_idempotent_connection_error_retries(self):
        assert _should_retry_connection_error("POST", True) is True


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestHookValidation:
    """Tests for FabricHook input validation."""

    def test_max_api_retries_zero_raises(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="validation_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        with pytest.raises(ValueError, match="max_api_retries must be >= 1"):
            FabricHook(fabric_conn_id="validation_conn", max_api_retries=0)

    def test_max_api_retries_negative_raises(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="validation_conn2",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        with pytest.raises(ValueError, match="max_api_retries must be >= 1"):
            FabricHook(fabric_conn_id="validation_conn2", max_api_retries=-1)

    def test_api_retry_delay_zero_raises(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="validation_conn3",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        with pytest.raises(ValueError, match="api_retry_delay must be >= 1"):
            FabricHook(fabric_conn_id="validation_conn3", api_retry_delay=0)

    def test_api_retry_delay_negative_raises(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="validation_conn4",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        with pytest.raises(ValueError, match="api_retry_delay must be >= 1"):
            FabricHook(fabric_conn_id="validation_conn4", api_retry_delay=-1)


class TestAsyncHookValidation:
    """Tests for FabricAsyncHook input validation (inherited from FabricHook)."""

    def test_async_max_api_retries_zero_raises(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="async_validation_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        with pytest.raises(ValueError, match="max_api_retries must be >= 1"):
            FabricAsyncHook(fabric_conn_id="async_validation_conn", max_api_retries=0)

    def test_async_api_retry_delay_zero_raises(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="async_validation_conn2",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        with pytest.raises(ValueError, match="api_retry_delay must be >= 1"):
            FabricAsyncHook(fabric_conn_id="async_validation_conn2", api_retry_delay=0)


# =============================================================================
# FabricRunItemException errorCode Tests
# =============================================================================

class TestFabricRunItemExceptionErrors:
    """Tests for the errorCode logic in get_item_run_details."""

    @pytest.fixture
    def hook(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="error_code_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        hook = FabricHook(fabric_conn_id="error_code_conn")
        hook._get_token = MagicMock(return_value="token")
        return hook

    def _mock_send_request(self, hook, json_data, mocker):
        response = MagicMock()
        response.ok = True
        response.json.return_value = json_data
        mocker.patch.object(hook, "_send_request", return_value=response)

    def test_raises_on_request_execution_failed(self, hook, mocker):
        self._mock_send_request(hook, {
            "status": "Failed",
            "failureReason": {"errorCode": "RequestExecutionFailed"},
        }, mocker)
        with pytest.raises(FabricRunItemException):
            hook.get_item_run_details(location=ITEM_RUN_LOCATION)

    def test_raises_on_not_found_error_code(self, hook, mocker):
        self._mock_send_request(hook, {
            "status": "Failed",
            "failureReason": {"errorCode": "NotFound"},
        }, mocker)
        with pytest.raises(FabricRunItemException):
            hook.get_item_run_details(location=ITEM_RUN_LOCATION)

    def test_no_raise_on_other_error_code(self, hook, mocker):
        data = {
            "status": "Failed",
            "failureReason": {"errorCode": "SomeOtherCode"},
        }
        self._mock_send_request(hook, data, mocker)
        result = hook.get_item_run_details(location=ITEM_RUN_LOCATION)
        assert result == data

    def test_no_raise_when_failure_reason_absent(self, hook, mocker):
        data = {"status": "Completed"}
        self._mock_send_request(hook, data, mocker)
        result = hook.get_item_run_details(location=ITEM_RUN_LOCATION)
        assert result == data

    def test_no_raise_when_failure_reason_is_none(self, hook, mocker):
        data = {"status": "Failed", "failureReason": None}
        self._mock_send_request(hook, data, mocker)
        result = hook.get_item_run_details(location=ITEM_RUN_LOCATION)
        assert result == data


# =============================================================================
# max_api_retries=1 Edge Case Tests
# =============================================================================

class TestMaxRetriesOneEdgeCase:
    """Tests that max_api_retries=1 means a single attempt with no retries."""

    @pytest.fixture
    def hook_single_attempt(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="single_attempt_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricHook(fabric_conn_id="single_attempt_conn", max_api_retries=1, api_retry_delay=1)

    @pytest.fixture
    def async_hook_single_attempt(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="async_single_attempt_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricAsyncHook(fabric_conn_id="async_single_attempt_conn", max_api_retries=1, api_retry_delay=1)

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_sync_max_api_retries_1_fails_immediately_on_503(self, mock_get, mock_sleep, mock_random, hook_single_attempt):
        """With max_api_retries=1, a retryable 503 raises immediately without sleeping."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 503
        error_response.headers = {}

        mock_get.return_value = error_response

        with pytest.raises(AirflowException, match="503"):
            hook_single_attempt._send_request("GET", "https://api.example.com/test")

        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_max_api_retries_1_fails_immediately_on_503(self, mock_session_class, mock_sleep, mock_random, async_hook_single_attempt):
        """With max_api_retries=1, async retryable 503 raises immediately without sleeping."""
        error_response = MagicMock()
        error_response.status = 503
        error_response.headers = {}

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=error_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="503"):
            await async_hook_single_attempt._async_send_request("GET", "https://api.example.com/test")

        assert mock_session.get.call_count == 1
        mock_sleep.assert_not_called()

    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.time.sleep")
    @patch(f"{MODULE}.requests.get")
    def test_sync_max_api_retries_1_fails_immediately_on_connection_error(self, mock_get, mock_sleep, mock_random, hook_single_attempt):
        """With max_api_retries=1, a ConnectionError raises immediately without sleeping."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(AirflowException, match="Connection refused"):
            hook_single_attempt._send_request("GET", "https://api.example.com/test")

        assert mock_get.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.random.uniform", return_value=0)
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_max_api_retries_1_fails_immediately_on_connection_error(self, mock_session_class, mock_sleep, mock_random, async_hook_single_attempt):
        """With max_api_retries=1, async ConnectionError raises immediately without sleeping."""
        mock_session = MagicMock()
        mock_session.get = AsyncMock(side_effect=aiohttp.ClientConnectionError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException, match="Connection refused"):
            await async_hook_single_attempt._async_send_request("GET", "https://api.example.com/test")

        assert mock_session.get.call_count == 1
        mock_sleep.assert_not_called()


# =============================================================================
# Async 202/204 No-Content Response Tests
# =============================================================================

class TestAsyncNoContentResponses:
    """Tests for _async_send_request handling of 202/204 responses."""

    @pytest.fixture
    def async_hook(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="no_content_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricAsyncHook(fabric_conn_id="no_content_conn")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_202_returns_empty_dict(self, mock_session_class, async_hook):
        """Test that HTTP 202 returns empty dict."""
        response = MagicMock()
        response.status = 202
        response.headers = {}

        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook._async_send_request("POST", "https://api.example.com/test")
        assert result == {}

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_204_returns_empty_dict(self, mock_session_class, async_hook):
        """Test that HTTP 204 returns empty dict."""
        response = MagicMock()
        response.status = 204
        response.headers = {}

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook._async_send_request("GET", "https://api.example.com/test")
        assert result == {}

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_202_with_json_body_parses_json(self, mock_session_class, async_hook):
        """Test that HTTP 202 with application/json Content-Type parses the JSON body."""
        response = MagicMock()
        response.status = 202
        response.headers = {"Content-Type": "application/json"}
        response.json = AsyncMock(return_value={"operationId": "abc-123", "status": "Running"})

        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook._async_send_request("POST", "https://api.example.com/test")
        assert result == {"operationId": "abc-123", "status": "Running"}

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_empty_content_type_returns_empty_dict(self, mock_session_class, async_hook):
        """Test that 200 with empty content-type returns empty dict."""
        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": ""}

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await async_hook._async_send_request("GET", "https://api.example.com/test")
        assert result == {}


# =============================================================================
# Token Refresh Retry Behavior Tests
# =============================================================================

class TestTokenRefreshRetry:
    """Tests for _get_token() and _async_get_token() retry behavior."""

    @pytest.fixture
    def hook(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="token_retry_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricHook(fabric_conn_id="token_retry_conn", max_api_retries=3, api_retry_delay=1)

    @pytest.fixture
    def async_hook(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="async_token_retry_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricAsyncHook(fabric_conn_id="async_token_retry_conn", max_api_retries=3, api_retry_delay=1)

    def test_get_token_passes_idempotent_and_is_token_request(self, hook, mocker):
        """Test that _get_token calls _send_request with idempotent=True and is_token_request=True."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        }
        mock_send = mocker.patch.object(hook, "_send_request", return_value=mock_response)
        mocker.patch(f"{MODULE}.update_conn")

        hook._get_token()

        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs.get("idempotent") is True
        assert kwargs.get("is_token_request") is True

    def test_get_token_wraps_failure_in_airflow_exception(self, hook, mocker):
        """Test that _get_token wraps _send_request failures."""
        mocker.patch.object(
            hook, "_send_request",
            side_effect=AirflowException("Request failed"),
        )

        with pytest.raises(AirflowException, match="Failed to refresh OAuth token"):
            hook._get_token()

    @pytest.mark.asyncio
    async def test_async_get_token_passes_idempotent_and_is_token_request(self, async_hook, mocker):
        """Test that _async_get_token calls _async_send_request with idempotent=True and is_token_request=True."""
        mock_response = {
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_in": 3600,
        }
        mock_send = mocker.patch.object(
            async_hook, "_async_send_request",
            return_value=mock_response,
        )
        mocker.patch(f"{MODULE}.update_conn")

        await async_hook._async_get_token()

        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs.get("idempotent") is True
        assert kwargs.get("is_token_request") is True

    @pytest.mark.asyncio
    async def test_async_get_token_wraps_failure_in_airflow_exception(self, async_hook, mocker):
        """Test that _async_get_token wraps _async_send_request failures."""
        mocker.patch.object(
            async_hook, "_async_send_request",
            side_effect=AirflowException("Request failed"),
        )

        with pytest.raises(AirflowException, match="Failed to refresh OAuth token"):
            await async_hook._async_get_token()


# =============================================================================
# Token Endpoint Error Sanitization Tests
# =============================================================================

class TestTokenEndpointErrorSanitization:
    """Tests that token endpoint errors don't leak response body."""

    @pytest.fixture
    def hook(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="sanitize_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricHook(fabric_conn_id="sanitize_conn", max_api_retries=1, api_retry_delay=1)

    @patch(f"{MODULE}.requests.post")
    def test_sync_token_error_omits_response_body(self, mock_post, hook):
        """Test that token endpoint errors don't include response body."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 400
        error_response.text = "sensitive OAuth error details"
        error_response.headers = {}
        mock_post.return_value = error_response

        with pytest.raises(AirflowException) as exc_info:
            hook._send_request(
                "POST",
                "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
                is_token_request=True,
            )

        assert "sensitive OAuth error details" not in str(exc_info.value)
        assert "Check connection credentials" in str(exc_info.value)

    @patch(f"{MODULE}.requests.get")
    def test_sync_non_token_error_includes_response_body(self, mock_get, hook):
        """Test that non-token endpoint errors still include response body."""
        error_response = MagicMock()
        error_response.ok = False
        error_response.status_code = 400
        error_response.text = "Bad Request details"
        error_response.headers = {}
        mock_get.return_value = error_response

        with pytest.raises(AirflowException) as exc_info:
            hook._send_request("GET", "https://api.fabric.microsoft.com/v1/test")

        assert "Bad Request details" in str(exc_info.value)

    @pytest.fixture
    def async_hook(self, create_mock_connection):
        create_mock_connection(
            Connection(
                conn_id="async_sanitize_conn",
                conn_type="generic",
                login="clientId",
                password="userRefreshToken",
                extra={"tenantId": "tenantId"},
            )
        )
        return FabricAsyncHook(fabric_conn_id="async_sanitize_conn", max_api_retries=1, api_retry_delay=1)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.aiohttp.ClientSession")
    async def test_async_token_error_omits_response_body(self, mock_session_class, async_hook):
        """Test that async token endpoint errors don't include response body."""
        error_response = MagicMock()
        error_response.status = 400
        error_response.headers = {}
        error_response.text = AsyncMock(return_value="sensitive OAuth error details")

        mock_session = MagicMock()
        mock_session.post = AsyncMock(return_value=error_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        with pytest.raises(AirflowException) as exc_info:
            await async_hook._async_send_request(
                "POST",
                "https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
                is_token_request=True,
            )

        assert "sensitive OAuth error details" not in str(exc_info.value)
        assert "Check connection credentials" in str(exc_info.value)
