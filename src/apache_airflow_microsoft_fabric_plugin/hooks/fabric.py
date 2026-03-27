from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import aiohttp
import httpx
import requests
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.models import Connection
from airflow.utils.session import provide_session

logger = logging.getLogger(__name__)

FABRIC_SCOPES = "https://api.fabric.microsoft.com/Item.Execute.All https://api.fabric.microsoft.com/Item.ReadWrite.All offline_access openid profile"

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Default wait time in seconds when Retry-After header is not present
DEFAULT_RETRY_AFTER_SECONDS = 5

# Absolute ceiling on server-specified Retry-After to prevent blocking a worker indefinitely
MAX_RETRY_AFTER_SECONDS = 300


def _get_retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """
    Extract wait time from Retry-After header.

    Handles both numeric seconds and HTTP-date formats per RFC 7231.
    Values exceeding MAX_RETRY_AFTER_SECONDS are clamped.

    :param headers: A dict-like object containing HTTP headers.
    :return: Number of seconds to wait before retrying, or None if header is absent.
    """
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None

    seconds: float | None = None
    try:
        seconds = max(float(retry_after), 0)
    except ValueError:
        pass

    if seconds is None:
        try:
            retry_date = parsedate_to_datetime(retry_after)
            seconds = max((retry_date - datetime.now(timezone.utc)).total_seconds(), 0)
        except (ValueError, TypeError):
            pass

    if seconds is None:
        logger.warning(
            "Could not parse Retry-After header value: %r. Using default of %s seconds.",
            retry_after,
            DEFAULT_RETRY_AFTER_SECONDS,
        )
        return DEFAULT_RETRY_AFTER_SECONDS

    if seconds > MAX_RETRY_AFTER_SECONDS:
        logger.warning(
            "Retry-After value %.1fs exceeds maximum of %ds; clamping.",
            seconds,
            MAX_RETRY_AFTER_SECONDS,
        )
        return float(MAX_RETRY_AFTER_SECONDS)

    return seconds


def _is_retryable_status_code(status_code: int) -> bool:
    """Check if the HTTP status code should trigger a retry."""
    return status_code in RETRYABLE_STATUS_CODES


def _calculate_backoff(
    retry_delay: float,
    attempt: int,
    retry_after: float | None = None,
    max_delay: float = 60.0,
) -> float:
    """
    Calculate wait time using exponential backoff with jitter.

    If a Retry-After value is provided (from an HTTP header), it is used as the
    floor for the delay. A small random jitter (0-25%) is added to prevent
    thundering herd.

    :param retry_delay: Base delay in seconds.
    :param attempt: Current attempt number (1-based).
    :param retry_after: Optional server-specified delay from Retry-After header.
    :param max_delay: Maximum allowed delay in seconds.
    :return: Seconds to wait before next retry.
    """
    exponential = retry_delay * (2 ** (attempt - 1))
    floor = retry_after if retry_after is not None else 0.0
    if retry_after is not None:
        delay = max(exponential, retry_after)
        effective_max = max(max_delay, retry_after)
    else:
        delay = exponential
        effective_max = max_delay
    delay = min(delay, effective_max)
    if delay >= effective_max:
        # At the cap: when the server specifies a Retry-After that exceeds
        # max_delay, the server's value is used exactly (no jitter) since the
        # server is authoritative. Otherwise, apply jitter via uniform(0.75, 1.0)
        # to provide thundering-herd protection at the cap.
        return max(effective_max * random.uniform(0.75, 1.0), floor)
    jitter = delay * random.uniform(0, 0.25)
    return min(max(delay + jitter, floor), effective_max)


def _should_retry_status(status_code: int, request_type: str, idempotent: bool) -> bool:
    """
    Determine whether a failed HTTP response should be retried.

    POST requests are only retried on 429 (rate limit) unless marked idempotent.
    GET requests (and idempotent POSTs) retry on all retryable status codes.

    :param status_code: The HTTP status code from the response.
    :param request_type: The HTTP method (uppercase), e.g. "GET", "POST".
    :param idempotent: Whether the request is safe to retry.
    :return: True if the request should be retried.
    """
    if not _is_retryable_status_code(status_code):
        return False
    if request_type == "POST" and not idempotent and status_code != 429:
        return False
    return True


def _should_retry_connection_error(request_type: str, idempotent: bool) -> bool:
    """
    Determine whether a connection/timeout error should be retried.

    POST requests are not retried on connection errors unless marked idempotent,
    because the server may have already processed the request.

    :param request_type: The HTTP method (uppercase).
    :param idempotent: Whether the request is safe to retry.
    :return: True if the request should be retried.
    """
    return request_type != "POST" or idempotent


@provide_session
def update_conn(conn_id, refresh_token: str, session=None):
    conn = session.query(Connection).filter(Connection.conn_id == conn_id).one()
    conn.password = refresh_token
    session.add(conn)
    session.commit()


class FabricRunItemStatus:
    """Fabric item run operation statuses."""

    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    NOT_STARTED = "NotStarted"
    DEDUPED = "Deduped"

    TERMINAL_STATUSES = {CANCELLED, FAILED, COMPLETED}
    INTERMEDIATE_STATES = {IN_PROGRESS}
    FAILURE_STATES = {FAILED, CANCELLED, DEDUPED}


class FabricRunItemException(AirflowException):
    """An exception that indicates a item run failed to complete."""


class FabricHook(BaseHook):
    """
    A hook to interact with Microsoft Fabric.
    This hook uses OAuth token generated from the refresh token, client ID and tenant ID specified in the connection.

    :param fabric_conn_id: Airflow Connection ID that contains the connection
        information for the Fabric account used for authentication.
    :param max_api_retries: Maximum number of attempts for transient HTTP errors (1 = single attempt,
        no retries). Defaults to 5.
    :param api_retry_delay: Initial delay in seconds between retries (exponential backoff). Defaults to 1.
    """  # noqa: D205

    conn_type: str = "fabric"
    conn_name_attr: str = "fabric_conn_id"
    default_conn_name: str = "fabric_default"
    hook_name: str = "MS Fabric"

    @classmethod
    def get_connection_form_widgets(cls) -> dict[str, Any]:
        """Return connection widgets to add to connection form."""
        from flask_appbuilder.fieldwidgets import BS3TextFieldWidget
        from flask_babel import lazy_gettext
        from wtforms import StringField

        return {
            "tenantId": StringField(lazy_gettext("Tenant ID"), widget=BS3TextFieldWidget()),
            "clientId": StringField(lazy_gettext("Client ID"), widget=BS3TextFieldWidget()),
        }

    @classmethod
    def get_ui_field_behaviour(cls) -> dict[str, Any]:
        """Return custom field behaviour."""
        return {
            "hidden_fields": ["schema", "port", "host", "extra"],
            "relabeling": {
                "login": "Client ID",
                "password": "Refresh Token",
            },
        }

    def __init__(
        self,
        *,
        fabric_conn_id: str = default_conn_name,
        max_api_retries: int = 5,
        api_retry_delay: int = 1
    ):
        if max_api_retries < 1:
            raise ValueError(f"max_api_retries must be >= 1 (got {max_api_retries})")
        if api_retry_delay < 1:
            raise ValueError(f"api_retry_delay must be >= 1 (got {api_retry_delay})")
        self.conn_id = fabric_conn_id
        self._api_version = "v1"
        self._base_url = "https://api.fabric.microsoft.com"
        self.max_api_retries = max_api_retries
        self.api_retry_delay = api_retry_delay
        self.cached_access_token: dict[str, str | None | int] = {"access_token": None, "expiry_time": 0}
        super().__init__()

    def _log_retry(
        self,
        url: str,
        status_or_error: int | str,
        attempt: int,
        wait_time: float,
    ) -> None:
        """Log a warning about an upcoming retry."""
        self.log.warning(
            "Request to %s failed with %s (attempt %d/%d). Retrying in %.1f seconds...",
            url,
            status_or_error,
            attempt,
            self.max_api_retries,
            wait_time,
        )

    def _log_exhausted(
        self,
        url: str,
        status_or_error: int | str,
    ) -> None:
        """Log an error when all retry attempts are exhausted."""
        self.log.error(
            "Request to %s failed with %s after %d attempts.",
            url,
            status_or_error,
            self.max_api_retries,
        )

    def _get_token(self) -> str:
        """
        If cached access token isn't expired, return it.

        Generate OAuth access token using refresh token in connection details and cache it.
        Update the connection with the new refresh token.

        :return: The access token.
        """
        access_token = self.cached_access_token.get("access_token")
        expiry_time = self.cached_access_token.get("expiry_time")

        if access_token and expiry_time > time.time():
            return str(access_token)

        connection = self.get_connection(self.conn_id)
        tenant_id = connection.extra_dejson.get("tenantId")
        client_id = connection.login
        client_secret = connection.extra_dejson.get("clientSecret")
        scopes = connection.extra_dejson.get("scopes", FABRIC_SCOPES)
        refresh_token = connection.password

        data = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
            "scope": scopes,
        }
        if client_secret:
            data["client_secret"] = client_secret

        try:
            # NOTE: Token refresh is marked idempotent for retry purposes. If the
            # refresh token has single-use rotation semantics (Azure AD default),
            # a retry after a network failure could send a now-stale refresh token.
            # This is an accepted risk because:
            # (1) Azure AD tolerates refresh token reuse within a short replay window.
            # (2) The alternative (no retry) is worse for transient network errors.
            response = self._send_request(
                "POST",
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                idempotent=True,
                is_token_request=True,
                data=data,
            )
        except AirflowException as e:
            raise AirflowException("Failed to refresh OAuth token. Check connection credentials.") from e

        response_data = response.json()

        api_access_token: str | None = response_data.get("access_token")
        api_refresh_token: str | None = response_data.get("refresh_token")

        if not api_access_token or not api_refresh_token:
            raise AirflowException("Failed to obtain access or refresh token from API.")

        update_conn(self.conn_id, api_refresh_token)

        self.cached_access_token = {
            "access_token": api_access_token,
            "expiry_time": time.time() + response_data.get("expires_in"),
        }

        return api_access_token

    def get_headers(self) -> dict[str, str]:
        """
        Form of auth headers based on OAuth token.

        :return: dict: Headers with the authorization token.
        """
        return {
            "Authorization": f"Bearer {self._get_token()}",
        }

    @staticmethod
    def _validate_item_run_details(item_run_details: dict) -> dict:
        """Raise if the item run details contain a known transient error code."""
        item_failure_reason = item_run_details.get("failureReason")
        if item_failure_reason is not None and item_failure_reason.get("errorCode") in [
            "RequestExecutionFailed", "NotFound"
        ]:
            raise FabricRunItemException("Unable to get item run details.")
        return item_run_details

    def get_item_run_details(self, location: str) -> dict:
        """
        Get details of the item run instance.

        :param location: The location of the item instance.
        :return: The item run details as a dictionary.
        """
        headers = self.get_headers()
        response = self._send_request("GET", location, headers=headers)
        return self._validate_item_run_details(response.json())

    def get_item_details(self, workspace_id: str, item_id: str) -> dict:
        """
        Get details of the item.

        :param workspace_id: The ID of the workspace in which the item is located.
        :param item_id: The ID of the item.

        :return: The details of the item.
        """
        url = f"{self._base_url}/{self._api_version}/workspaces/{workspace_id}/items/{item_id}"

        headers = self.get_headers()
        response = self._send_request("GET", url, headers=headers)
        return response.json()


    def run_fabric_item(self, workspace_id: str, item_id: str, job_type: str, job_params: dict = None, config: dict = None) -> str:
        """
        Run a Fabric item.

        :param workspace_id: The workspace Id in which the item is located.
        :param item_id: The item Id. To check available items, Refer to: https://learn.microsoft.com/rest/api/fabric/admin/items/list-items?tabs=HTTP#itemtype.
        :param job_type: The type of job to run. For running a notebook, this should be "RunNotebook".
        :param job_params: An optional dictionary of parameters to pass to the job.

        :return: The run Id of item.
        """
        url = f"{self._base_url}/{self._api_version}/workspaces/{workspace_id}/items/{item_id}/jobs/instances?jobType={job_type}"

        headers = self.get_headers()

        data = {}
        if job_params or config:
            data["executionData"] = {
                "parameters": job_params or {},
                "configuration": config or {},
        }

        # NOTE: This POST is intentionally NOT marked idempotent. Only 429 (rate limit)
        # is retried for non-idempotent POSTs. The Fabric API returns 429 before
        # processing the request, so retrying is safe and won't create duplicate runs.
        response = self._send_request("POST", url, headers=headers, json=data)

        location_header = response.headers.get("Location")
        if location_header is None:
            raise AirflowException("Location header not found in run on demand item response.")

        return location_header

    # TODO: output value from notebook should be available in xcom - not available in API yet

    def wait_for_item_run_status(
        self,
        location: str,
        target_status: str,
        check_interval: int = 60,
        timeout: int = 60 * 60 * 24 * 7,
    ) -> bool:
        """
        Wait for the item run to reach a target status.

        :param location: The location of the item instance retrieved from the header of item run API.
        :param target_status: The status to wait for.
        :param check_interval: The interval at which to check the status.
        :param timeout: The maximum time to wait for the status.

        :return: True if the item run reached the target status, False otherwise.
        """
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            item_run_details = self.get_item_run_details(location)
            item_run_status = item_run_details["status"]
            if item_run_status in FabricRunItemStatus.TERMINAL_STATUSES:
                return item_run_status == target_status
            self.log.info("Sleeping for %s. The pipeline state is %s.", check_interval, item_run_status)
            time.sleep(check_interval)
        raise FabricRunItemException(
            f"Item run did not reach the target status {target_status} within the {timeout} seconds."
        )

    def _send_request(
        self, request_type: str, url: str, *, idempotent: bool = False, is_token_request: bool = False, **kwargs
    ) -> requests.Response:
        """
        Send a request to the REST API with automatic retry for transient errors.

        Handles rate limits (429) and server errors (5xx) with exponential backoff.
        Respects the Retry-After header when present.

        :param request_type: The type of the request (GET, POST).
        :param url: The URL against which the request needs to be made.
        :param idempotent: Whether this request is safe to retry on server errors and connection failures.
        :param is_token_request: Whether this is an OAuth token request. When True, error messages
            avoid including the response body to prevent leaking sensitive token data.
        :param kwargs: Additional keyword arguments to be passed to the request function.
        :return: The response object returned by the request.
        :raises AirflowException: If the request fails after all retries or a non-retryable error occurs.
        """
        request_funcs: dict[str, Callable[..., requests.Response]] = {
            "GET": requests.get,
            "POST": requests.post,
        }

        request_type_upper = request_type.upper()
        func = request_funcs.get(request_type_upper)
        if func is None:
            raise AirflowException(f"Unsupported request type: {request_type}")

        for attempt in range(1, self.max_api_retries + 1):
            try:
                response = func(url=url, **kwargs)

                if response.ok:
                    return response

                if _should_retry_status(response.status_code, request_type_upper, idempotent):
                    retry_after = _get_retry_after_seconds(response.headers)
                    if attempt < self.max_api_retries:
                        wait_time = _calculate_backoff(self.api_retry_delay, attempt, retry_after=retry_after)
                        self._log_retry(url, f"status {response.status_code}", attempt, wait_time)
                        time.sleep(wait_time)
                        continue
                    else:
                        self._log_exhausted(url, f"status {response.status_code}")
                        raise AirflowException(
                            f"Request to {url} failed with status {response.status_code} "
                            f"after {self.max_api_retries} attempts."
                        )

                # Non-retryable error - raise immediately
                if is_token_request:
                    raise AirflowException(
                        f"Token request failed with status {response.status_code}. "
                        "Check connection credentials."
                    )
                raise AirflowException(
                    f"Request to {url} failed with status {response.status_code}: {response.text[:500]}"
                )

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if not _should_retry_connection_error(request_type_upper, idempotent):
                    raise AirflowException(
                        f"Request to {url} failed: {e}. "
                        "WARNING: The server may have already processed this request."
                    ) from e
                if attempt < self.max_api_retries:
                    wait_time = _calculate_backoff(self.api_retry_delay, attempt)
                    self._log_retry(url, type(e).__name__, attempt, wait_time)
                    time.sleep(wait_time)
                else:
                    self._log_exhausted(url, type(e).__name__)
                    raise AirflowException(
                        f"Request to {url} failed after {self.max_api_retries} attempts: {e}"
                    ) from e

        # Defensive: should never be reached; loop always returns or raises
        raise AirflowException(
            f"Request to {url} failed after {self.max_api_retries} attempts."
        )


class FabricAsyncHook(FabricHook):
    """
    Interact with Microsoft Fabric asynchronously.

    :param fabric_conn_id: Airflow Connection ID that contains the connection.
    :param max_api_retries: Maximum number of attempts for transient HTTP errors (1 = single attempt,
        no retries). Defaults to 5.
    :param api_retry_delay: Initial delay in seconds between retries. Defaults to 1.
    """

    default_conn_name: str = "fabric_default"

    def __init__(
        self,
        *,
        fabric_conn_id: str = default_conn_name,
        max_api_retries: int = 5,
        api_retry_delay: int = 1,
    ):
        super().__init__(fabric_conn_id=fabric_conn_id, max_api_retries=max_api_retries, api_retry_delay=api_retry_delay)

    async def _async_send_request(
        self, request_type: str, url: str, *, idempotent: bool = False, is_token_request: bool = False, **kwargs
    ) -> Any:
        """
        Asynchronously sends a HTTP request with automatic retry for transient errors.

        Handles rate limits (429) and server errors (5xx) with exponential backoff.
        Respects the Retry-After header when present.

        :param request_type: The HTTP method to use ('GET', 'POST', etc.).
        :param url: The URL to send the request to.
        :param idempotent: Whether this request is safe to retry on server errors and connection failures.
        :param is_token_request: Whether this is an OAuth token request. When True, error messages
            avoid including the response body to prevent leaking sensitive token data.
        :param kwargs: Additional arguments to pass to the request method.
        :return: The response from the server.
        :raises AirflowException: If the request fails after all retries.
        """
        request_type_upper = request_type.upper()

        async with aiohttp.ClientSession() as session:
            if request_type_upper == "GET":
                request_func = session.get
            elif request_type_upper == "POST":
                request_func = session.post
            else:
                raise AirflowException(f"Unsupported request type: {request_type}")

            for attempt in range(1, self.max_api_retries + 1):
                try:
                    response = await request_func(url, **kwargs)

                    # Success path
                    if response.status < 400:
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'application/json' in content_type:
                            return await response.json()
                        elif 'application/octet-stream' in content_type:
                            return await response.read()
                        elif response.status in (202, 204) or not content_type:
                            return {}
                        else:
                            raise AirflowException(f"Unsupported Content-Type: {content_type}")

                    if _should_retry_status(response.status, request_type_upper, idempotent):
                        retry_after = _get_retry_after_seconds(response.headers)
                        if attempt < self.max_api_retries:
                            wait_time = _calculate_backoff(self.api_retry_delay, attempt, retry_after=retry_after)
                            self._log_retry(url, f"status {response.status}", attempt, wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            self._log_exhausted(url, f"status {response.status}")
                            raise AirflowException(
                                f"Request to {url} failed with status {response.status} "
                                f"after {self.max_api_retries} attempts."
                            )

                    # Non-retryable error - raise immediately
                    if is_token_request:
                        raise AirflowException(
                            f"Token request failed with status {response.status}. "
                            "Check connection credentials."
                        )
                    error_text = await response.text()
                    raise AirflowException(
                        f"Request to {url} failed with status {response.status}: {error_text[:500]}"
                    )

                except (aiohttp.ClientConnectionError, aiohttp.ClientOSError, asyncio.TimeoutError) as e:
                    if not _should_retry_connection_error(request_type_upper, idempotent):
                        raise AirflowException(
                            f"Request to {url} failed: {e}. "
                            "WARNING: The server may have already processed this request."
                        ) from e
                    if attempt < self.max_api_retries:
                        wait_time = _calculate_backoff(self.api_retry_delay, attempt)
                        self._log_retry(url, type(e).__name__, attempt, wait_time)
                        await asyncio.sleep(wait_time)
                    else:
                        self._log_exhausted(url, type(e).__name__)
                        raise AirflowException(f"Request to {url} failed: {e}") from e

        # Defensive: should never be reached; loop always returns or raises
        raise AirflowException(
            f"Async request to {url} failed after {self.max_api_retries} attempts."
        )

    async def _async_get_token(self) -> str:
        """
        Get the access token from the refresh token.

        :return: The access token.
        """
        cached_token = self.cached_access_token.get("access_token")
        expiry_time = self.cached_access_token.get("expiry_time")

        if isinstance(cached_token, str) and isinstance(expiry_time, float) and expiry_time > time.time():
            return str(cached_token)

        connection = await asyncio.to_thread(self.get_connection, self.conn_id)
        tenant_id = connection.extra_dejson.get("tenantId")
        client_id = connection.login
        client_secret = connection.extra_dejson.get("clientSecret")
        refresh_token = connection.password
        scopes = connection.extra_dejson.get("scopes", FABRIC_SCOPES)

        data = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
            "scope": scopes,
        }
        if client_secret:
            data["client_secret"] = client_secret

        try:
            # NOTE: Token refresh is marked idempotent for retry purposes. If the
            # refresh token has single-use rotation semantics (Azure AD default),
            # a retry after a network failure could send a now-stale refresh token.
            # This is an accepted risk because:
            # (1) Azure AD tolerates refresh token reuse within a short replay window.
            # (2) The alternative (no retry) is worse for transient network errors.
            response = await self._async_send_request(
                "POST",
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
                idempotent=True,
                is_token_request=True,
                data=data,
            )
        except AirflowException as e:
            raise AirflowException("Failed to refresh OAuth token. Check connection credentials.") from e
        api_access_token: str | None = response.get("access_token")
        api_refresh_token: str | None = response.get("refresh_token")

        if not api_access_token or not api_refresh_token:
            raise AirflowException("Failed to obtain access or refresh token from API.")

        await asyncio.to_thread(update_conn, self.conn_id, api_refresh_token)

        self.cached_access_token = {
            "access_token": api_access_token,
            "expiry_time": time.time() + response.get("expires_in"),
        }

        return api_access_token

    async def async_get_headers(self) -> dict[str, str]:
        """
        Form of auth headers based on OAuth token.

        :return: dict: Headers with the authorization token.
        """
        access_token = await self._async_get_token()

        return {
            "Authorization": f"Bearer {access_token}",
        }

    async def async_get_item_run_details(self, workspace_id: str, item_id: str, item_run_id: str) -> dict:
        """
        Get run details of the item instance.

        :param workspace_id: The workspace ID.
        :param item_id: The item ID.
        :param item_run_id: The item run ID.
        :return: The item run details as a dictionary.
        """
        url = f"{self._base_url}/{self._api_version}/workspaces/{workspace_id}/items/{item_id}/jobs/instances/{item_run_id}"
        headers = await self.async_get_headers()
        response = await self._async_send_request("GET", url, headers=headers)
        return self._validate_item_run_details(response)

    async def cancel_item_run(self, workspace_id: str, item_id: str, item_run_id: str):
        """
        Cancel the item run.

        :param workspace_id: The workspace Id in which the item is located.
        :param item_id: The item Id.
        :param item_run_id: The Id of the item run.

        """
        url = f"{self._base_url}/{self._api_version}/workspaces/{workspace_id}/items/{item_id}/jobs/instances/{item_run_id}/cancel"
        headers = await self.async_get_headers()
        return await self._async_send_request("POST", url, headers=headers, idempotent=True)
