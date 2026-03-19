from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from airflow.exceptions import AirflowException
from airflow.triggers.base import BaseTrigger, TriggerEvent
from apache_airflow_microsoft_fabric_plugin.hooks.fabric import FabricAsyncHook, FabricRunItemStatus


class FabricTrigger(BaseTrigger):
    """Trigger when a Fabric item run finishes.

    :param fabric_conn_id: Airflow Connection ID for Fabric.
    :param item_run_id: The ID of the running item.
    :param workspace_id: The workspace ID.
    :param item_id: The item ID.
    :param job_type: The job type (e.g., RunNotebook, Pipeline).
    :param end_time: The time at which to stop polling (monotonic clock).
    :param check_interval: Seconds between status checks. Defaults to 60.
    :param wait_for_termination: Whether to wait for the job to finish. Defaults to True.
    :param max_api_retries: Maximum number of attempts for transient HTTP errors (1 = single attempt,
        no retries). Defaults to 5.
    :param api_retry_delay: Initial delay in seconds between retries. Defaults to 1.
    """

    def __init__(
        self,
        fabric_conn_id: str,
        item_run_id: str,
        workspace_id: str,
        item_id: str,
        job_type: str,
        end_time: float,
        check_interval: int = 60,
        wait_for_termination: bool = True,
        max_api_retries: int = 5,
        api_retry_delay: int = 1,
    ):
        super().__init__()
        self.fabric_conn_id = fabric_conn_id
        self.item_run_id = item_run_id
        self.workspace_id = workspace_id
        self.item_id = item_id
        self.job_type = job_type
        self.end_time = end_time
        self.check_interval = check_interval
        self.wait_for_termination = wait_for_termination
        self.max_api_retries = max_api_retries
        self.api_retry_delay = api_retry_delay

    def serialize(self):
        """Serialize the FabricTrigger instance."""
        return (
            "apache_airflow_microsoft_fabric_plugin.triggers.fabric.FabricTrigger",
            {
                "fabric_conn_id": self.fabric_conn_id,
                "item_run_id": self.item_run_id,
                "workspace_id": self.workspace_id,
                "item_id": self.item_id,
                "job_type": self.job_type,
                "end_time": self.end_time,
                "check_interval": self.check_interval,
                "wait_for_termination": self.wait_for_termination,
                "max_api_retries": self.max_api_retries,
                "api_retry_delay": self.api_retry_delay,
            },
        )

    async def run(self) -> AsyncIterator[TriggerEvent]:
        """Make async connection to the fabric and polls for the item run status."""
        hook = FabricAsyncHook(
            fabric_conn_id=self.fabric_conn_id,
            max_api_retries=self.max_api_retries,
            api_retry_delay=self.api_retry_delay,
        )

        try:
            item_run_status = None
            while self.end_time > time.time():
                item_run_details = await hook.async_get_item_run_details(
                    item_run_id=self.item_run_id,
                    workspace_id=self.workspace_id,
                    item_id=self.item_id,
                )
                item_run_status = item_run_details["status"]
                if item_run_status == FabricRunItemStatus.COMPLETED:
                    yield TriggerEvent(
                        {
                            "status": "success",
                            "message": f"The item run {self.item_run_id} has status {item_run_status}.",
                            "run_id": self.item_run_id,
                            "item_run_status": item_run_status,
                        }
                    )
                    return
                elif item_run_status in FabricRunItemStatus.FAILURE_STATES:
                    yield TriggerEvent(
                        {
                            "status": "error",
                            "message": f"The item run {self.item_run_id} has status {item_run_status}.",
                            "run_id": self.item_run_id,
                            "item_run_status": item_run_status,
                        }
                    )
                    return

                self.log.info(
                    "Sleeping for %s. The item state is %s.",
                    self.check_interval,
                    item_run_status,
                )
                await asyncio.sleep(self.check_interval)
            # Timeout reached
            if item_run_status is not None:
                message = f"Timeout reached: The item run {self.item_run_id} has {item_run_status}."
            else:
                message = f"Timeout reached: The item run {self.item_run_id} status is unknown."

            yield TriggerEvent(
                {
                    "status": "error",
                    "message": message,
                    "run_id": self.item_run_id,
                    "item_run_status": item_run_status,
                }
            )
        except AirflowException as error:
            try:
                self.log.error(
                    "Unexpected error %s caught. Cancel pipeline run %s",
                    error,
                    self.item_run_id,
                )
                await hook.cancel_item_run(
                    item_run_id=self.item_run_id,
                    workspace_id=self.workspace_id,
                    item_id=self.item_id,
                )
                yield TriggerEvent(
                    {
                        "status": "error",
                        "message": str(error),
                        "run_id": self.item_run_id,
                        "item_run_status": FabricRunItemStatus.CANCELLED,
                    }
                )
                return
            except Exception as error:
                yield TriggerEvent(
                    {
                        "status": "error",
                        "message": str(error),
                        "run_id": self.item_run_id,
                        "item_run_status": "unknown",
                    }
                )
                return
