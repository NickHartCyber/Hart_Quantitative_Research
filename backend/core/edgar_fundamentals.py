from __future__ import annotations

import logging
import math
from typing import Any

from backend.platform_apis.edgar_api import edgar_api as edgar

log = logging.getLogger("edgar_fundamentals")

_EDGAR_QUARTERLY_FPS = {"Q1", "Q2", "Q3", "Q4"}
_EDGAR_ANNUAL_FPS = {"FY"}
_EDGAR_ANNUAL_FORMS = ("10-K", "20-F", "40-F")

_EDGAR_TAGS_REVENUE = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
_EDGAR_TAGS_EPS = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "EarningsPerShareBasic",
)
_EDGAR_TAGS_GROSS_PROFIT = ("GrossProfit",)
_EDGAR_TAGS_NET_INCOME = (
    "NetIncomeLoss",
    "ProfitLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
)
_EDGAR_TAGS_OCF = (
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
)
_EDGAR_TAGS_CAPEX = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "CapitalExpenditures",
)
_EDGAR_TAGS_CASH = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "CashAndCashEquivalents",
)
_EDGAR_TAGS_DEBT_LT = (
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "DebtAndCapitalLeaseObligations",
    "Debt",
)
_EDGAR_TAGS_DEBT_ST = ("DebtCurrent", "ShortTermBorrowings", "ShortTermDebt")
_EDGAR_TAGS_EQUITY = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
_EDGAR_TAGS_EBITDA = ("EBITDA", "Ebitda")
_EDGAR_TAGS_SHARES = ("CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding")

_EDGAR_UNIT_USD = ("USD",)
_EDGAR_UNIT_USD_PER_SHARE = ("USD/shares", "USD / shares")
_EDGAR_UNIT_SHARES = ("shares", "Shares")


def _safe_number(val: Any) -> float | None:
    try:
        num = float(val)
    except Exception:
        return None
    return num if math.isfinite(num) else None


def _edgar_pick_units(units: dict[str, Any], preferred: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(units, dict) or not units:
        return []
    for unit in preferred:
        values = units.get(unit)
        if isinstance(values, list):
            return values
    for values in units.values():
        if isinstance(values, list):
            return values
    return []


def _edgar_clean_items(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not items:
        return []
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        end = str(item.get("end") or "").strip()
        if not end:
            continue
        val = _safe_number(item.get("val"))
        if val is None:
            continue
        cleaned.append(
            {
                "end": end,
                "val": val,
                "fp": str(item.get("fp") or "").upper(),
                "form": str(item.get("form") or "").upper(),
                "filed": str(item.get("filed") or ""),
            }
        )
    if not cleaned:
        return []
    by_end: dict[str, dict[str, Any]] = {}
    for item in cleaned:
        end = item["end"]
        prev = by_end.get(end)
        if prev is None or (item["filed"] and item["filed"] > prev.get("filed", "")):
            by_end[end] = item
    return sorted(by_end.values(), key=lambda row: row["end"])


def _edgar_get_fact_items(
    facts: dict[str, Any],
    tags: tuple[str, ...],
    *,
    unit_preference: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not isinstance(facts, dict):
        return []
    fact_root = facts.get("facts") if isinstance(facts.get("facts"), dict) else {}
    for tag in tags:
        for taxonomy in ("us-gaap", "dei"):
            if taxonomy not in fact_root:
                continue
            entry = fact_root.get(taxonomy, {}).get(tag)
            if not isinstance(entry, dict):
                continue
            units = entry.get("units", {})
            items = _edgar_pick_units(units, unit_preference)
            cleaned = _edgar_clean_items(items)
            if cleaned:
                return cleaned
    return []


def _edgar_quarterly_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in items if row.get("fp") in _EDGAR_QUARTERLY_FPS]


def _edgar_annual_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annual: list[dict[str, Any]] = []
    for row in items:
        fp = row.get("fp") or ""
        form = row.get("form") or ""
        if fp in _EDGAR_ANNUAL_FPS or any(form.startswith(prefix) for prefix in _EDGAR_ANNUAL_FORMS):
            annual.append(row)
    return annual


def _edgar_preferred_period(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_items = _edgar_quarterly_items(items)
    if len(q_items) >= 4:
        return q_items
    a_items = _edgar_annual_items(items)
    if a_items:
        return a_items
    return items


def _edgar_last_value(items: list[dict[str, Any]]) -> float | None:
    return items[-1]["val"] if items else None


def _edgar_ttm_sum(items: list[dict[str, Any]]) -> float | None:
    q_items = _edgar_quarterly_items(items)
    if len(q_items) >= 4:
        return sum(row["val"] for row in q_items[-4:])
    a_items = _edgar_annual_items(items)
    if a_items:
        return a_items[-1]["val"]
    return None


def _edgar_ttm_growth(items: list[dict[str, Any]]) -> float | None:
    q_items = _edgar_quarterly_items(items)
    if len(q_items) >= 8:
        recent = sum(row["val"] for row in q_items[-4:])
        prev = sum(row["val"] for row in q_items[-8:-4])
        if prev:
            return (recent - prev) / abs(prev)
    a_items = _edgar_annual_items(items)
    if len(a_items) >= 2:
        recent = a_items[-1]["val"]
        prev = a_items[-2]["val"]
        if prev:
            return (recent - prev) / abs(prev)
    return None


def _edgar_yoy_change(items: list[dict[str, Any]]) -> float | None:
    q_items = _edgar_quarterly_items(items)
    if len(q_items) >= 5:
        recent = q_items[-1]["val"]
        prev = q_items[-5]["val"]
        if prev:
            return (recent - prev) / abs(prev)
    a_items = _edgar_annual_items(items)
    if len(a_items) >= 2:
        recent = a_items[-1]["val"]
        prev = a_items[-2]["val"]
        if prev:
            return (recent - prev) / abs(prev)
    return None


def _edgar_margin_series(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
) -> list[float]:
    if not numerator or not denominator:
        return []
    num_map = {row["end"]: row["val"] for row in numerator}
    den_map = {row["end"]: row["val"] for row in denominator}
    margins: list[float] = []
    for end in sorted(set(num_map) & set(den_map)):
        den = den_map[end]
        if den:
            margins.append(num_map[end] / den)
    return margins


def _edgar_fcf_series(
    ocf_items: list[dict[str, Any]],
    capex_items: list[dict[str, Any]],
) -> list[float]:
    if not ocf_items:
        return []
    ocf_map = {row["end"]: row["val"] for row in ocf_items}
    capex_map = {row["end"]: row["val"] for row in capex_items} if capex_items else {}
    series: list[float] = []
    for end in sorted(ocf_map):
        capex = capex_map.get(end, 0.0)
        series.append(ocf_map[end] + capex)
    return series


def get_edgar_fundamentals(
    ticker: str,
    *,
    price_hint: float | None = None,
    include_filings: bool = True,
    timeout: float | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Pull fundamentals from SEC EDGAR (no external fallbacks).
    Returns (fundamentals_dict, source label).
    """
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return {}, "edgar"

    cik = edgar.resolve_cik_from_ticker(symbol, timeout=timeout)
    if not cik:
        log.warning("EDGAR CIK lookup failed for %s", symbol)
        return {}, "edgar"

    facts = edgar.get_company_facts(cik, timeout=timeout)
    submissions = edgar.get_company_submissions(cik, timeout=timeout) if include_filings else None

    revenue_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_REVENUE, unit_preference=_EDGAR_UNIT_USD)
    eps_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_EPS, unit_preference=_EDGAR_UNIT_USD_PER_SHARE)
    gross_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_GROSS_PROFIT, unit_preference=_EDGAR_UNIT_USD)
    net_income_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_NET_INCOME, unit_preference=_EDGAR_UNIT_USD)
    ocf_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_OCF, unit_preference=_EDGAR_UNIT_USD)
    capex_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_CAPEX, unit_preference=_EDGAR_UNIT_USD)
    cash_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_CASH, unit_preference=_EDGAR_UNIT_USD)
    debt_lt_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_DEBT_LT, unit_preference=_EDGAR_UNIT_USD)
    debt_st_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_DEBT_ST, unit_preference=_EDGAR_UNIT_USD)
    equity_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_EQUITY, unit_preference=_EDGAR_UNIT_USD)
    ebitda_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_EBITDA, unit_preference=_EDGAR_UNIT_USD)
    shares_items = _edgar_get_fact_items(facts, _EDGAR_TAGS_SHARES, unit_preference=_EDGAR_UNIT_SHARES)

    revenue_ttm = _edgar_ttm_sum(revenue_items)
    revenue_growth = _edgar_ttm_growth(revenue_items)
    eps_ttm = _edgar_ttm_sum(eps_items)
    eps_growth = _edgar_yoy_change(eps_items)
    net_income_ttm = _edgar_ttm_sum(net_income_items)

    gross_pref = _edgar_preferred_period(gross_items)
    revenue_pref = _edgar_preferred_period(revenue_items)
    margin_series = _edgar_margin_series(gross_pref, revenue_pref)
    gross_margin_latest = margin_series[-1] if margin_series else None
    gross_margin_trend = None
    if len(margin_series) >= 4:
        gross_margin_trend = margin_series[-1] - sum(margin_series[-4:-1]) / 3.0

    ocf_pref = _edgar_preferred_period(ocf_items)
    capex_pref = _edgar_preferred_period(capex_items) if capex_items else []
    fcf_series = _edgar_fcf_series(ocf_pref, capex_pref)
    fcf_ttm = None
    if len(fcf_series) >= 4:
        fcf_ttm = sum(fcf_series[-4:])
    elif fcf_series:
        fcf_ttm = fcf_series[-1]

    cash_latest = _edgar_last_value(cash_items) or 0.0
    debt_latest = (_edgar_last_value(debt_lt_items) or 0.0) + (_edgar_last_value(debt_st_items) or 0.0)
    net_debt = debt_latest - cash_latest if debt_latest else None

    equity_latest = _edgar_last_value(equity_items)
    ebitda_latest = _edgar_last_value(ebitda_items)

    shares_latest = _edgar_last_value(shares_items)
    dilution_rate = None
    if shares_items and shares_items[0]["val"] > 0:
        dilution_rate = (shares_items[-1]["val"] - shares_items[0]["val"]) / shares_items[0]["val"]

    price = price_hint if price_hint is not None else None
    market_cap = price * shares_latest if price and shares_latest else None
    book_value = (equity_latest / shares_latest) if equity_latest and shares_latest else None
    ps_ratio = (market_cap / revenue_ttm) if market_cap and revenue_ttm else None
    pe_ratio = (price / eps_ttm) if price and eps_ttm else None
    roe = (net_income_ttm / equity_latest) if net_income_ttm and equity_latest else None
    debt_to_equity = (debt_latest / equity_latest) if debt_latest and equity_latest else None
    net_debt_to_ebitda = (
        (net_debt / ebitda_latest) if net_debt is not None and ebitda_latest and ebitda_latest > 0 else None
    )
    enterprise_value = (market_cap + max(net_debt or 0.0, 0.0)) if market_cap is not None else None
    ev_to_ebitda = (
        (enterprise_value / ebitda_latest) if enterprise_value and ebitda_latest and ebitda_latest > 0 else None
    )
    fcf_margin = (fcf_ttm / revenue_ttm) if fcf_ttm and revenue_ttm else None
    fcf_yield = (fcf_ttm / market_cap) if fcf_ttm and market_cap else None

    filings: list[dict[str, Any]] = []
    sic_desc = None
    if isinstance(submissions, dict):
        sic_desc = submissions.get("sicDescription") or submissions.get("sic")
        filings = edgar.extract_company_filings(submissions)
        filings = [
            row
            for row in filings
            if edgar.is_form_variant(row.get("form"), edgar.FUNDAMENTAL_FORM_BASES)
        ]
        filings.sort(key=lambda row: str(row.get("filingDate") or ""), reverse=True)

    fundamentals: dict[str, Any] = {
        "ticker": symbol,
        "cik": cik,
        "name": facts.get("entityName") if isinstance(facts, dict) else None,
        "sector": sic_desc or None,
        "currency": "USD",
        "price": price,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "forward_pe": None,
        "peg_ratio": None,
        "eps_ttm": eps_ttm,
        "roe": roe,
        "debt_to_equity": debt_to_equity,
        "dividend_yield": None,
        "profit_margin": (net_income_ttm / revenue_ttm) if net_income_ttm and revenue_ttm else None,
        "book_value": book_value,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "beta": None,
        "shares_outstanding": shares_latest,
        "revenue_ttm": revenue_ttm,
        "revenue_growth": revenue_growth,
        "eps_growth": eps_growth,
        "gross_margin": gross_margin_latest,
        "gross_margin_trend": gross_margin_trend,
        "fcf_ttm": fcf_ttm,
        "fcf_margin": fcf_margin,
        "fcf_yield": fcf_yield,
        "net_debt": net_debt,
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "ps_ratio": ps_ratio,
        "ev_to_ebitda": ev_to_ebitda,
        "inst_percent": None,
        "inst_trending_up": None,
        "dilution_rate": dilution_rate,
        "filings": filings if include_filings else [],
    }

    return fundamentals, "edgar"
