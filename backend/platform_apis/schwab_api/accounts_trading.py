"""
Accounts & Trading utilities for the Schwab Trader API (Retail).

This module wraps a few common account and order endpoints:
- Retrieve the account's opaque hash value (required by other endpoints)
- Read account balances (and persist a CSV snapshot)
- Compute settled funds
- List recent orders (for today) or by an arbitrary time range
- Place and replace orders

References
----------
Schwab Trader API (Retail) docs:
https://developer.schwab.com/products/trader-api--individual/details/specifications/Retail%20Trader%20API%20Production

Notes
-----
* This code assumes an external helper `retrieve_auth_token_value()` that returns a
  dict with a valid OAuth `access_token` (and handles refresh outside this class).
* Timestamps passed to the orders endpoints are expected in **UTC** ISO 8601 with
  fractional seconds, e.g. "2025-08-20T14:23:45.000Z".
* Several methods persist CSV snapshots under the configured `DATA_DIR`
  (see `backend.core.config.data_path`).
* For client-side correlation, pass a header "Schwab-Client-CorrelId" on
  POST/PUT order requests. Use the optional `headers` param on `post_order`
  and `replace_order` to supply it.
"""

from __future__ import annotations

import datetime
import json

import pandas as pd
import requests
from loguru import logger

from backend.core.config import data_path

from .retrieve_secrets_and_tokens import retrieve_auth_token_value


class AccountsTrading:
    """
    Thin client for Schwab account + order operations.

    This client:
      * lazily refreshes an access token on instantiation,
      * resolves the account's opaque `hashValue`,
      * exposes helpers to query balances and orders,
      * and provides simple order placement/replacement methods.

    Attributes
    ----------
    access_token : str | None
        Current OAuth access token used in the Authorization header.
    account_hash_value : str | None
        Opaque account hash required by Trader API endpoints.
    base_url : str
        Base URL for the Schwab Trader API.
    headers : dict[str, str]
        Default request headers containing the bearer token.
    """

    def __init__(self) -> None:
        """
        Create a new client and pre-load authentication + account identifiers.

        Side Effects
        ------------
        - Retrieves and stores an access token via `retrieve_auth_token_value()`.
        - Resolves and stores `account_hash_value` from `/accounts/accountNumbers`.
        """
        self.access_token: str | None = None
        self.account_hash_value: str | None = None

        self.base_url = "https://api.schwabapi.com/trader/v1"
        self.headers = {}  # filled after token refresh

        self.refresh_access_token()
        self.get_account_number_hash_value()

    def refresh_access_token(self) -> None:
        """
        Refresh the in-memory OAuth access token and Authorization header.

        Notes
        -----
        This relies on an external token store/flow implemented in
        `retrieve_auth_token_value()`. If your token can expire, call this
        again before requests that might 401.
        """
        token_payload = retrieve_auth_token_value()
        self.access_token = token_payload["access_token"]
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    def get_account_number_hash_value(self) -> None:
        """
        Resolve and cache the account `hashValue` needed for most account endpoints.

        Endpoint
        --------
        GET /accounts/accountNumbers

        Raises
        ------
        requests.HTTPError
            If the HTTP request fails (non-2xx) or response parsing fails.
        """
        resp = requests.get(f"{self.base_url}/accounts/accountNumbers", headers=self.headers)
        resp.raise_for_status()
        frame = pd.json_normalize(resp.json())
        self.account_hash_value = frame["hashValue"].iloc[0]

    def get_account_balances(self) -> pd.DataFrame:
        """
        Fetch account balances and persist a CSV snapshot to `DATA_DIR`.

        Endpoint
        --------
        GET /accounts/{accountHash}

        Returns
        -------
        pandas.DataFrame
            Normalized response containing the `securitiesAccount` structure.

        Side Effects
        ------------
        - Writes `account_balances.csv` into the configured data directory.

        Raises
        ------
        requests.HTTPError
            If the HTTP request fails.
        """
        assert self.account_hash_value, "Account hash value not loaded."
        resp = requests.get(
            f"{self.base_url}/accounts/{self.account_hash_value}",
            headers=self.headers,
            params={"fields": "positions"},  # include positions payload for holdings/PnL
        )
        resp.raise_for_status()
        frame = pd.json_normalize(resp.json())
        frame.to_csv(data_path("account_balances.csv"), index=False)
        return frame

    def get_settled_funds(self) -> int | float:
        """
        Compute settled cash available for trading (settled minus unsettled).

        Returns
        -------
        int | float
            The settled cash value derived from the balances payload.

        Notes
        -----
        This calls `get_account_balances()` under the hood and uses:
        - `securitiesAccount.currentBalances.cashAvailableForTrading`
        - `securitiesAccount.currentBalances.unsettledCash`
        """
        balances = self.get_account_balances()
        settled_cash = (
            balances.loc[0, "securitiesAccount.currentBalances.cashAvailableForTrading"]
            - balances.loc[0, "securitiesAccount.currentBalances.unsettledCash"]
        )
        return settled_cash

    def get_recent_orders(self) -> pd.DataFrame | str:
        """
        Retrieve *today's* orders (UTC midnight → now) and persist a dated CSV.

        Endpoint
        --------
        GET /accounts/{accountHash}/orders

        Returns
        -------
        pandas.DataFrame | str
            DataFrame of recent orders when successful; otherwise raw error text
            from the API for 400 responses.

        Side Effects
        ------------
        - Writes `YYYY-MM-DD_orders.csv` under `DATA_DIR`.

        Notes
        -----
        Schwab expects UTC timestamps in the query. We compute this range as:
        - from: today's UTC midnight (00:00:00.000Z)
        - to: now (UTC)
        """
        today_date = datetime.date.today()
        midnight_utc = datetime.datetime.combine(today_date, datetime.time(0, 0, 0))
        from_entered = midnight_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        to_entered = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        resp = requests.get(
            f"{self.base_url}/accounts/{self.account_hash_value}/orders",
            params={"fromEnteredTime": from_entered, "toEnteredTime": to_entered},
            headers=self.headers,
        )

        if resp.status_code == 200:
            data = resp.json()
            frame = pd.DataFrame(data)
            frame.to_csv(data_path(f"{today_date}_orders.csv"), index=False)
            return frame
        elif resp.status_code == 400:
            logger.error(resp.text)
            return resp.text
        else:
            # Surface unexpected statuses to logs for observability
            logger.error(f"Unexpected status: {resp.status_code} | body={resp.text}")
            resp.raise_for_status()
            return resp.text  # pragma: no cover

    def get_orders_by_date(self, from_time: str, to_time: str) -> pd.DataFrame | str:
        """
        Retrieve orders between two UTC ISO-8601 timestamps.

        Endpoint
        --------
        GET /accounts/{accountHash}/orders

        Parameters
        ----------
        from_time : str
            Start timestamp in UTC ISO format (e.g. "2025-08-20T00:00:00.000Z").
        to_time : str
            End timestamp in UTC ISO format.

        Returns
        -------
        pandas.DataFrame | str
            DataFrame when successful; otherwise the error text for 400 responses.

        Raises
        ------
        requests.HTTPError
            For non-400, non-200 responses.
        """
        resp = requests.get(
            f"{self.base_url}/accounts/{self.account_hash_value}/orders",
            params={"fromEnteredTime": from_time, "toEnteredTime": to_time},
            headers=self.headers,
        )

        if resp.status_code == 200:
            data = resp.json()
            return pd.DataFrame(data)
        elif resp.status_code == 400:
            logger.error(resp.text)
            return resp.text
        else:
            logger.error(f"Unexpected status: {resp.status_code} | body={resp.text}")
            resp.raise_for_status()
            return resp.text  # pragma: no cover

    def post_order(self, order_payload: dict, headers: dict | None = None) -> dict:
        """
        Submit a new order.

        Endpoint
        --------
        POST /accounts/{accountHash}/orders

        Parameters
        ----------
        order_payload : dict
            JSON-serializable order body per Schwab spec.
        headers : dict | None, optional
            Extra HTTP headers to merge into the request. This is the preferred way
            to supply a client correlation id, e.g.:
                {"Schwab-Client-CorrelId": "<your-correlation-id>"}

        Returns
        -------
        dict | None
            The submitted payload when a 201 is returned; otherwise `None`.

        Logging
        -------
        - Logs the API response body and a success message for 201s.
        - Logs error text and code for non-201 statuses.
        """
        logger.debug(order_payload)
        payload_str = json.dumps(order_payload)

        # Base headers (auth + JSON). Merge any caller-supplied headers on top.
        request_headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(headers)

        resp = requests.post(
            f"{self.base_url}/accounts/{self.account_hash_value}/orders",
            data=payload_str,      # keep explicit JSON string; alternatively use json=order_payload
            headers=request_headers,
        )

        if resp.status_code == 201:
            # logger.info(resp.text)
            logger.info("Order Placed Successfully")
            return order_payload

        logger.error(f"Error in response with message: {resp.text}")
        logger.error(f"Error code: {resp.status_code}")
        return resp.text

    def replace_order(self, order_payload: dict, order_id: int | str, headers: dict | None = None) -> dict:
        """
        Replace an existing order by ID.

        Endpoint
        --------
        PUT /accounts/{accountHash}/orders/{orderId}

        Parameters
        ----------
        order_payload : dict
            Replacement order body.
        order_id : int | str
            The Schwab order identifier.
        headers : dict | None, optional
            Extra HTTP headers to merge into the request. This is the preferred way
            to supply a client correlation id for the replace call, e.g.:
                {"Schwab-Client-CorrelId": "<your-correlation-id>"}

        Returns
        -------
        dict | None
            The submitted payload when a 201 is returned; otherwise `None`.

        Logging
        -------
        - Logs the API response body and success message for 201s.
        - Logs error text and code for non-201 statuses.
        """
        payload_str = json.dumps(order_payload)

        request_headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if headers:
            request_headers.update(headers)

        resp = requests.put(
            f"{self.base_url}/accounts/{self.account_hash_value}/orders/{order_id}",
            data=payload_str,      # alternatively: json=order_payload
            headers=request_headers,
        )

        if resp.status_code == 201:
            logger.info(resp.text)
            logger.info("Order Placed Successfully")
            return order_payload

        logger.error(f"Error in response with message: {resp.text}")
        logger.error(f"Error code: {resp.status_code}")
        return resp.text

    def deduct_buy_from_settled_funds(self, order_payload: dict) -> None:
        """
        Adjust the cached balances CSV to reflect a buy order's unsettled cash impact.

        This is a **local bookkeeping utility** that:
          1) Re-reads the account balances via `get_account_balances()`,
          2) Computes used cash as `quantity * price`,
          3) Increases `currentBalances.unsettledCash` by that amount,
          4) Persists the updated balances CSV.

        Parameters
        ----------
        order_payload : dict
            Must contain numeric `quantity` and `price` fields.

        Side Effects
        ------------
        - Overwrites `account_balances.csv` in `DATA_DIR`.

        Notes
        -----
        This does *not* update balances on Schwab; it only updates your local CSV snapshot
        to reflect the expected unsettled cash change.
        """
        balances = self.get_account_balances()
        used_cash = order_payload["quantity"] * order_payload["price"]

        unsettled_cash = balances.loc[0, "securitiesAccount.currentBalances.unsettledCash"]
        balances.loc[0, "securitiesAccount.currentBalances.unsettledCash"] = (
            used_cash + unsettled_cash
        )

        balances.to_csv(data_path("account_balances.csv"), index=False)
