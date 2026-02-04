import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Link,
  MenuItem,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import { History } from "@mui/icons-material";
import HistoryCalendarDropdown from "../components/HistoryCalendarDropdown";

const API_BASE = process.env.REACT_APP_API_BASE || "";
const MASS_COMBO_LABEL = "Mass combo list";
const DEFAULT_LIMIT = 10;
const DEFAULT_STRIKE_COUNT = 8;
const HISTORY_LATEST = "latest";

const SUGGESTER_CHOICES = [
  {
    key: "short_term",
    label: "1-2 day momentum",
    description: "Breakout-oriented entries sized for a quick 1-2 day hold.",
  },
  {
    key: "swing_term",
    label: "3-4 month swing",
    description: "Medium-term trend and 3-6 month momentum with wider stops/targets.",
  },
  {
    key: "long_term",
    label: "1+ year compounder",
    description: "Long-term trend, positive 12m returns, and lower realized volatility bias.",
  },
];

function formatCurrency(value) {
  if (!Number.isFinite(Number(value))) return "--";
  return `$${Number(value).toFixed(2)}`;
}

function formatNumber(value, digits = 0) {
  if (!Number.isFinite(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return "--";
  const num = Number(value);
  const ratio = Math.abs(num) <= 1 ? num * 100 : num;
  return `${num >= 0 ? "+" : ""}${ratio.toFixed(1)}%`;
}

function formatPercentChange(value) {
  if (!Number.isFinite(Number(value))) return "--";
  const num = Number(value);
  return `${num >= 0 ? "+" : ""}${num.toFixed(2)}%`;
}

function formatSignedCurrencyChange(value) {
  if (!Number.isFinite(Number(value))) return "--";
  const abs = Math.abs(Number(value));
  return `${value >= 0 ? "+" : "-"}${formatCurrency(abs)}`;
}

function renderTickerLink(value) {
  const ticker = String(value || "").trim();
  if (!ticker) return "--";
  return (
    <Link component={RouterLink} to={`/ticker?ticker=${encodeURIComponent(ticker)}`} underline="hover" color="inherit">
      {ticker}
    </Link>
  );
}

function formatExpiration(expiration, days) {
  if (expiration) {
    const parsed = new Date(expiration);
    if (!Number.isNaN(parsed.valueOf())) {
      const label = parsed.toLocaleDateString(undefined, { month: "short", day: "numeric" });
      return days ? `${label} (${days}d)` : label;
    }
  }
  if (Number.isFinite(Number(days))) {
    return `${days}d`;
  }
  return "--";
}

function formatHistoryLabel(entry) {
  const label = entry?.label || entry?.date || "";
  const count = Number(entry?.count);
  if (!label) return "Unknown run";
  if (Number.isFinite(count)) return `${label} (${count} idea${count === 1 ? "" : "s"})`;
  return label;
}

function formatHistoryDateLabel(dateStr) {
  if (typeof dateStr !== "string") return "";
  if (dateStr.length === 8) {
    return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6)}`;
  }
  return dateStr;
}

function normalizeHistoryDateKey(value) {
  if (typeof value !== "string") {
    return "";
  }
  return value.replace(/-/g, "").trim();
}

function getOldestHistoryDate(entries) {
  const keys = (entries || [])
    .map((entry) => normalizeHistoryDateKey(entry?.date))
    .filter((key) => key.length === 8)
    .sort();
  if (!keys.length) {
    return "";
  }
  return formatHistoryDateLabel(keys[0]);
}

function formatHistoryViewLabel(label, count) {
  if (!label) return "History run";
  const num = Number(count);
  if (Number.isFinite(num)) {
    return `History: ${label} (${num} idea${num === 1 ? "" : "s"})`;
  }
  return `History: ${label}`;
}

function normalizeOptionSymbol(value) {
  return String(value || "").replace(/\s+/g, "").toUpperCase();
}

function pickOptionRecommendationPrice(option) {
  if (!option) return null;
  const mid = Number(option.mid);
  if (Number.isFinite(mid)) return mid;
  const bid = Number(option.bid);
  const ask = Number(option.ask);
  if (Number.isFinite(bid) && Number.isFinite(ask)) return (bid + ask) / 2;
  const mark = Number(option.mark);
  if (Number.isFinite(mark)) return mark;
  const last = Number(option.last);
  if (Number.isFinite(last)) return last;
  return null;
}

function pickOptionCurrentPrice(quote) {
  if (!quote) return null;
  const mid = Number(quote.mid);
  if (Number.isFinite(mid) && mid > 0) return mid;
  const bid = Number(quote.bid);
  const ask = Number(quote.ask);
  if (Number.isFinite(bid) && Number.isFinite(ask) && bid > 0 && ask > 0) return (bid + ask) / 2;
  const mark = Number(quote.mark);
  if (Number.isFinite(mark) && mark > 0) return mark;
  const last = Number(quote.last);
  if (Number.isFinite(last) && last > 0) return last;
  const close = Number(quote.close);
  if (Number.isFinite(close) && close > 0) return close;
  if (Number.isFinite(mid)) return mid;
  if (Number.isFinite(mark)) return mark;
  if (Number.isFinite(last)) return last;
  if (Number.isFinite(close)) return close;
  return null;
}

function formatOptionContract(option) {
  if (!option) return "--";
  const type = option.put_call ? option.put_call.toUpperCase() : "OPTION";
  const strike = Number.isFinite(Number(option.strike)) ? formatCurrency(option.strike) : "--";
  const expiration = formatExpiration(option.expiration, option.days_to_expiration);
  return `${type} ${strike} @ ${expiration}`;
}

function OptionSummary({ option }) {
  if (!option) {
    return (
      <Typography variant="body2" color="text.secondary">
        No contract picked
      </Typography>
    );
  }
  const hasSpreadPct = Number.isFinite(Number(option.spread_pct));
  const hasSpread = Number.isFinite(Number(option.spread));
  const hasBreakEven = Number.isFinite(Number(option.break_even));
  const hasMoneyness = Number.isFinite(Number(option.moneyness_pct));

  return (
    <Stack spacing={0.25}>
      <Typography variant="body2" fontWeight={700}>
        {formatCurrency(option.strike)} @ {formatExpiration(option.expiration, option.days_to_expiration)}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Mid {formatCurrency(option.mid)} • Spread{" "}
        {hasSpreadPct ? formatPercent(option.spread_pct) : hasSpread ? formatCurrency(option.spread) : "--"}
        {Number.isFinite(Number(option.target_dte)) ? ` • Target ${option.target_dte}d` : ""}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        OI {formatNumber(option.open_interest)} • Vol {formatNumber(option.volume)} • Δ{" "}
        {formatNumber(option.delta, 2)} • IV {formatPercent(option.implied_volatility)}
      </Typography>
      {hasBreakEven || hasMoneyness ? (
        <Typography variant="caption" color="text.secondary">
          {hasBreakEven ? `Breakeven ${formatCurrency(option.break_even)}` : ""}
          {hasBreakEven && hasMoneyness ? " • " : ""}
          {hasMoneyness ? `Moneyness ${formatPercent(option.moneyness_pct)}` : ""}
        </Typography>
      ) : null}
    </Stack>
  );
}

function OptionsTable({ rows, includePuts }) {
  if (!rows || rows.length === 0) {
    return <Alert severity="info">No cached options ideas yet.</Alert>;
  }

  return (
    <Box sx={{ overflowX: "auto" }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Ticker</TableCell>
            <TableCell>Underlying</TableCell>
            <TableCell>Call idea</TableCell>
            {includePuts ? <TableCell>Put idea</TableCell> : null}
            <TableCell>Notes</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.ticker}>
              <TableCell>
                <Stack spacing={0.25}>
                  <Typography variant="body2" fontWeight={700}>
                    {renderTickerLink(row.ticker)}
                  </Typography>
                  {row.action ? (
                    <Typography variant="caption" color="text.secondary">
                      {row.action}
                    </Typography>
                  ) : null}
                </Stack>
              </TableCell>
              <TableCell>
                <Stack spacing={0.25}>
                  <Typography variant="body2">Px: {formatCurrency(row.underlying_price)}</Typography>
                  {row.equity_entry ? (
                    <Typography variant="caption" color="text.secondary">
                      Entry ref: {formatCurrency(row.equity_entry)}
                    </Typography>
                  ) : null}
                </Stack>
              </TableCell>
              <TableCell>
                <OptionSummary option={row.best_call} />
              </TableCell>
              {includePuts ? (
                <TableCell>
                  <OptionSummary option={row.best_put} />
                </TableCell>
              ) : null}
              <TableCell>
                {row.chain_error ? (
                  <Typography variant="caption" color="error.main" display="block">
                    {row.chain_error}
                  </Typography>
                ) : null}
                {row.warnings && row.warnings.length > 0 ? (
                  <Typography variant="caption" color="text.secondary" display="block">
                    {row.warnings.join(" • ")}
                  </Typography>
                ) : null}
                <Typography variant="caption" color="text.secondary" display="block">
                  Calls checked: {row.call_candidates ?? 0}
                  {includePuts ? ` • Puts checked: ${row.put_candidates ?? 0}` : ""}
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

function OptionsPerformanceTable({ rows, loading, error, viewLabel, lastRun, controls = null }) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const totals = safeRows.reduce(
    (acc, row) => {
      const pct = Number(row.percentChange);
      const dollar = Number(row.dollarChange);
      if (Number.isFinite(pct)) {
        acc.totalPercentChange += pct;
        acc.percentCount += 1;
      }
      if (Number.isFinite(dollar)) {
        acc.totalDollarChange += dollar;
        acc.dollarCount += 1;
      }
      return acc;
    },
    { totalPercentChange: 0, percentCount: 0, totalDollarChange: 0, dollarCount: 0 },
  );
  const totalPercentChange = totals.percentCount > 0 ? totals.totalPercentChange : null;
  const totalDollarChange = totals.dollarCount > 0 ? totals.totalDollarChange : null;
  const totalPctColor =
    totalPercentChange > 0 ? "success.main" : totalPercentChange < 0 ? "error.main" : "text.primary";
  const totalDollarColor =
    totalDollarChange > 0 ? "success.main" : totalDollarChange < 0 ? "error.main" : "text.primary";

  return (
    <Paper variant="outlined" sx={{ p: 2.5, display: "grid", gap: 1.5 }}>
      <Stack
        direction={{ xs: "column", md: "row" }}
        alignItems={{ xs: "flex-start", md: "center" }}
        justifyContent="space-between"
        spacing={1}
      >
        <Box>
          <Typography variant="h6">Performance since recommendation</Typography>
          <Typography variant="body2" color="text.secondary">
            Select a saved run and compare each contract&apos;s mid price to today&apos;s quote.
          </Typography>
          {viewLabel ? (
            <Typography variant="caption" color="text.secondary">
              Viewing: {viewLabel}
            </Typography>
          ) : null}
          {lastRun ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
              Run timestamp: {lastRun}
            </Typography>
          ) : null}
        </Box>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ xs: "flex-start", sm: "center" }}>
          {controls}
          {loading ? <CircularProgress size={22} /> : null}
        </Stack>
      </Stack>
      {error ? <Alert severity="error">{error}</Alert> : null}
      {!loading && safeRows.length === 0 ? (
        <Alert severity="info">Select a run to see performance.</Alert>
      ) : (
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="right">% Change since Recommendation</TableCell>
                <TableCell align="right">$ Change per Contract since Recommendation</TableCell>
                <TableCell>Ticker</TableCell>
                <TableCell>Contract</TableCell>
                <TableCell>Action</TableCell>
                <TableCell align="right">Price at recommendation</TableCell>
                <TableCell align="right">Current price</TableCell>
                <TableCell>Horizon</TableCell>
                <TableCell>Thesis</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {safeRows.map((row, idx) => {
                const pct = Number(row.percentChange);
                const dollar = Number(row.dollarChange);
                const pctColor = pct > 0 ? "success.main" : pct < 0 ? "error.main" : "text.primary";
                const dollarColor =
                  dollar > 0 ? "success.main" : dollar < 0 ? "error.main" : "text.primary";
                const symbolLabel = row.optionSymbol || row.optionDescription;
                return (
                  <TableRow key={`${row.optionSymbol || row.ticker || idx}-${idx}`}>
                    <TableCell align="right" sx={{ color: pctColor }}>
                      {formatPercentChange(row.percentChange)}
                    </TableCell>
                    <TableCell align="right" sx={{ color: dollarColor }}>
                      {formatSignedCurrencyChange(row.dollarChange)}
                    </TableCell>
                    <TableCell>{renderTickerLink(row.ticker)}</TableCell>
                    <TableCell>
                      <Stack spacing={0.25}>
                        <Typography variant="body2" fontWeight={700}>
                          {formatOptionContract(row.option)}
                        </Typography>
                        {symbolLabel ? (
                          <Typography variant="caption" color="text.secondary">
                            {symbolLabel}
                          </Typography>
                        ) : null}
                      </Stack>
                    </TableCell>
                    <TableCell>{row.action || "--"}</TableCell>
                    <TableCell align="right">{formatCurrency(row.recommendationPrice)}</TableCell>
                    <TableCell align="right">{formatCurrency(row.currentPrice)}</TableCell>
                    <TableCell>{row.horizon || "--"}</TableCell>
                    <TableCell sx={{ maxWidth: 420 }}>{row.thesis || "--"}</TableCell>
                  </TableRow>
                );
              })}
              <TableRow>
                <TableCell align="right" sx={{ fontWeight: 700, color: totalPctColor }}>
                  {formatPercentChange(totalPercentChange)}
                </TableCell>
                <TableCell align="right" sx={{ fontWeight: 700, color: totalDollarColor }}>
                  {formatSignedCurrencyChange(totalDollarChange)}
                </TableCell>
                <TableCell colSpan={7} sx={{ fontWeight: 700 }}>
                  Totals
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </Box>
      )}
    </Paper>
  );
}

export default function OptionsSuggesterPage() {
  const [activeHorizon, setActiveHorizon] = useState(SUGGESTER_CHOICES[0].key);
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [strikeCount, setStrikeCount] = useState(DEFAULT_STRIKE_COUNT);
  const [includePuts, setIncludePuts] = useState(true);
  const [optionsRows, setOptionsRows] = useState([]);
  const [lastRun, setLastRun] = useState(null);
  const [viewLabel, setViewLabel] = useState("Latest run");
  const [performanceSuggestions, setPerformanceSuggestions] = useState([]);
  const [performanceViewLabel, setPerformanceViewLabel] = useState("Latest run");
  const [performanceLastRun, setPerformanceLastRun] = useState(null);
  const [performanceRunLoading, setPerformanceRunLoading] = useState(false);
  const [performanceRunError, setPerformanceRunError] = useState("");
  const [performanceHistorySelected, setPerformanceHistorySelected] = useState(HISTORY_LATEST);
  const [optionQuoteLookup, setOptionQuoteLookup] = useState({});
  const [performanceError, setPerformanceError] = useState("");
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [error, setError] = useState("");
  const [historyRuns, setHistoryRuns] = useState([]);
  const [historySelected, setHistorySelected] = useState(HISTORY_LATEST);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [historyRunLoading, setHistoryRunLoading] = useState(false);

  const activeMeta = useMemo(
    () => SUGGESTER_CHOICES.find((choice) => choice.key === activeHorizon) || SUGGESTER_CHOICES[0],
    [activeHorizon],
  );
  const oldestHistoryDate = useMemo(() => getOldestHistoryDate(historyRuns), [historyRuns]);
  const historyHelperText = historyLoading
    ? "Loading saved runs..."
    : historyRuns.length > 0
    ? `${historyRuns.length} saved run${historyRuns.length === 1 ? "" : "s"}${oldestHistoryDate ? ` | Oldest: ${oldestHistoryDate}` : ""}`
    : "No saved runs yet";
  const performanceContracts = useMemo(() => {
    const rows = Array.isArray(performanceSuggestions) ? performanceSuggestions : [];
    const contracts = [];
    rows.forEach((row) => {
      const ticker = String(row?.ticker || "").trim().toUpperCase();
      const base = {
        ticker,
        action: row?.action || row?.signal,
        horizon: row?.horizon || activeHorizon,
        thesis: row?.thesis || row?.reason,
      };
      if (row?.best_call) {
        contracts.push({ ...base, option: row.best_call, optionType: "CALL" });
      }
      if (row?.best_put) {
        contracts.push({ ...base, option: row.best_put, optionType: "PUT" });
      }
    });
    return contracts;
  }, [activeHorizon, performanceSuggestions]);
  const performanceRows = useMemo(
    () =>
      performanceContracts.map((entry) => {
        const option = entry.option || {};
        const optionSymbol = String(option.symbol || "").trim();
        const optionDescription = String(option.description || "").trim();
        const optionSymbolKey = normalizeOptionSymbol(optionSymbol);
        const quote = optionSymbolKey ? optionQuoteLookup[optionSymbolKey] : null;
        const recommendationPrice = pickOptionRecommendationPrice(option);
        const currentPrice = pickOptionCurrentPrice(quote);
        const dollarChange =
          Number.isFinite(recommendationPrice) && Number.isFinite(currentPrice)
            ? currentPrice - recommendationPrice
            : null;
        const percentChange =
          Number.isFinite(dollarChange) && Number.isFinite(recommendationPrice) && recommendationPrice !== 0
            ? (dollarChange / recommendationPrice) * 100
            : null;

        return {
          ...entry,
          option,
          optionSymbol,
          optionDescription,
          recommendationPrice,
          currentPrice,
          dollarChange,
          percentChange,
        };
      }),
    [optionQuoteLookup, performanceContracts],
  );
  const performanceHistoryHelperText =
    performanceHistorySelected === HISTORY_LATEST
      ? "Compare performance using the latest loaded suggestions"
      : `Performance for ${formatHistoryDateLabel(performanceHistorySelected)}`;
  const combinedPerformanceError = [performanceRunError, performanceError].filter(Boolean).join(" | ");
  const performanceLoadingCombined = performanceLoading || performanceRunLoading;
  const performanceHistoryValue = performanceHistorySelected || HISTORY_LATEST;

  const fetchHistoryRuns = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const res = await fetch(`${API_BASE}/api/suggest/options/history?horizon=${activeHorizon}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const runs = Array.isArray(data?.history) ? data.history : [];
      setHistoryRuns(runs);
      return runs;
    } catch (e) {
      setHistoryError(e.message || String(e));
      setHistoryRuns([]);
      return [];
    } finally {
      setHistoryLoading(false);
    }
  }, [activeHorizon]);

  const loadLatestHistory = useCallback(async () => {
    setHistoryRunLoading(true);
    setError("");
    try {
      const runs = await fetchHistoryRuns();
      if (!runs.length) {
        setOptionsRows([]);
        setLastRun(null);
        setViewLabel("Latest run");
        return;
      }
      const latest = runs[0];
      const params = new URLSearchParams({
        horizon: activeHorizon,
        limit: String(limit),
        strike_count: String(strikeCount),
        include_puts: includePuts ? "true" : "false",
      });
      const res = await fetch(`${API_BASE}/api/suggest/options/history/${latest.date}?${params.toString()}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setOptionsRows(Array.isArray(data?.options) ? data.options : []);
      setLastRun(data?.last_run || data?.date || null);
      const label = data?.label || formatHistoryDateLabel(latest.date);
      setViewLabel(label ? `Latest: ${label}` : "Latest run");
      setHistorySelected(HISTORY_LATEST);
      setPerformanceHistorySelected(HISTORY_LATEST);
    } catch (e) {
      setError(e.message || String(e));
      setOptionsRows([]);
    } finally {
      setHistoryRunLoading(false);
    }
  }, [activeHorizon, fetchHistoryRuns, includePuts, limit, strikeCount]);

  const loadHistoryRun = useCallback(
    async () => {
      if (!historySelected || historySelected === HISTORY_LATEST) {
        return loadLatestHistory();
      }
      setHistoryRunLoading(true);
      setError("");
      try {
        const params = new URLSearchParams({
          horizon: activeHorizon,
          limit: String(limit),
          strike_count: String(strikeCount),
          include_puts: includePuts ? "true" : "false",
        });
        const res = await fetch(`${API_BASE}/api/suggest/options/history/${historySelected}?${params.toString()}`);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        setOptionsRows(Array.isArray(data?.options) ? data.options : []);
        setLastRun(data?.last_run || data?.date || null);
        const label = data?.label || formatHistoryDateLabel(historySelected);
        setViewLabel(label ? `History: ${label}` : "History run");
        setPerformanceHistorySelected(HISTORY_LATEST);
      } catch (e) {
        setError(e.message || String(e));
        setOptionsRows([]);
      } finally {
        setHistoryRunLoading(false);
      }
    },
    [activeHorizon, historySelected, includePuts, limit, loadLatestHistory, strikeCount],
  );

  useEffect(() => {
    setHistorySelected(HISTORY_LATEST);
    setViewLabel("Latest run");
    setPerformanceHistorySelected(HISTORY_LATEST);
    setPerformanceViewLabel("Latest run");
  }, [activeHorizon]);

  useEffect(() => {
    if (historySelected === HISTORY_LATEST) {
      loadLatestHistory();
    }
  }, [activeHorizon, historySelected, includePuts, limit, loadLatestHistory, strikeCount]);

  useEffect(() => {
    setPerformanceRunError("");
    setPerformanceError("");
  }, [activeHorizon]);

  useEffect(() => {
    if (performanceHistorySelected !== HISTORY_LATEST) {
      return;
    }
    setPerformanceSuggestions(optionsRows);
    setPerformanceViewLabel(viewLabel);
    setPerformanceLastRun(lastRun);
    setPerformanceRunError("");
    setPerformanceRunLoading(false);
  }, [lastRun, optionsRows, performanceHistorySelected, viewLabel]);

  useEffect(() => {
    if (!performanceHistorySelected || performanceHistorySelected === HISTORY_LATEST) {
      return undefined;
    }

    let cancelled = false;
    setPerformanceRunLoading(true);
    setPerformanceRunError("");
    setPerformanceViewLabel(formatHistoryViewLabel(formatHistoryDateLabel(performanceHistorySelected)));
    setPerformanceLastRun(null);

    async function loadPerformanceHistory() {
      try {
        const params = new URLSearchParams({
          horizon: activeHorizon,
          limit: String(limit),
          strike_count: String(strikeCount),
          include_puts: includePuts ? "true" : "false",
        });
        const res = await fetch(
          `${API_BASE}/api/suggest/options/history/${performanceHistorySelected}?${params.toString()}`,
        );
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const items = Array.isArray(data?.options) ? data.options : [];
        const label = data?.label || formatHistoryDateLabel(performanceHistorySelected);
        const count = data?.count ?? items.length;
        if (cancelled) return;
        setPerformanceSuggestions(items);
        setPerformanceViewLabel(formatHistoryViewLabel(label, count));
        setPerformanceLastRun(data?.last_run || data?.date || null);
      } catch (e) {
        if (!cancelled) {
          setPerformanceRunError(e.message || String(e));
          setPerformanceSuggestions([]);
          setPerformanceViewLabel(formatHistoryViewLabel(formatHistoryDateLabel(performanceHistorySelected)));
          setPerformanceLastRun(null);
        }
      } finally {
        if (!cancelled) {
          setPerformanceRunLoading(false);
        }
      }
    }

    loadPerformanceHistory();

    return () => {
      cancelled = true;
    };
  }, [activeHorizon, includePuts, limit, performanceHistorySelected, strikeCount]);

  useEffect(() => {
    const rawSymbols = (performanceSuggestions || [])
      .flatMap((row) => [row?.best_call?.symbol, row?.best_put?.symbol])
      .map((value) => String(value || "").trim())
      .filter((value) => value);
    const symbolMap = new Map();
    rawSymbols.forEach((symbol) => {
      const key = normalizeOptionSymbol(symbol);
      if (key && !symbolMap.has(key)) {
        symbolMap.set(key, symbol);
      }
    });
    const symbols = Array.from(symbolMap.values());

    if (symbols.length === 0) {
      setOptionQuoteLookup({});
      setPerformanceError("");
      setPerformanceLoading(false);
      return;
    }

    let cancelled = false;
    setPerformanceLoading(true);
    setPerformanceError("");

    fetch(`${API_BASE}/api/quotes?symbols=${encodeURIComponent(symbols.join(","))}`)
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const next = {};
        const quotes = Array.isArray(data?.quotes) ? data.quotes : [];
        quotes.forEach((quote) => {
          const symbol = String(quote?.symbol || "").trim();
          const key = normalizeOptionSymbol(symbol);
          if (key) {
            next[key] = quote;
          }
        });
        setOptionQuoteLookup(next);
        const missing = Array.isArray(data?.missing) ? data.missing : [];
        if (missing.length > 0) {
          setPerformanceError(`Missing quotes for ${missing.join(", ")}`);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setPerformanceError(err.message || String(err));
          setOptionQuoteLookup({});
        }
      })
      .finally(() => {
        if (!cancelled) {
          setPerformanceLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [performanceSuggestions]);

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 2.5 }}>
      <Stack
        direction={{ xs: "column", md: "row" }}
        alignItems={{ xs: "flex-start", md: "center" }}
        justifyContent="space-between"
        spacing={1.5}
      >
        <Box>
          <Typography variant="h5">Options Ideas</Typography>
          <Typography variant="body2" color="text.secondary">
            View the latest cached options ideas generated by the scheduled stock ideas refresh. Uses the {MASS_COMBO_LABEL}.
          </Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap" }}>
            {SUGGESTER_CHOICES.map((choice) => (
              <Chip
                key={choice.key}
                label={choice.label}
                color={choice.key === activeHorizon ? "primary" : "default"}
                variant={choice.key === activeHorizon ? "filled" : "outlined"}
                onClick={() => setActiveHorizon(choice.key)}
                sx={{ mr: 1, mb: 1 }}
              />
            ))}
          </Stack>
          {activeMeta.description ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
              {activeMeta.description}
            </Typography>
          ) : null}
          {lastRun ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
              Last run: {lastRun}
            </Typography>
          ) : null}
        </Box>
      </Stack>

      {error && <Alert severity="error">{error}</Alert>}

      <Paper variant="outlined" sx={{ p: 2.5, display: "grid", gap: 1.5 }}>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={1}
          alignItems={{ xs: "flex-start", md: "center" }}
          justifyContent="space-between"
        >
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ xs: "flex-start", sm: "center" }}>
            <TextField
              type="number"
              size="small"
              label="Ticker limit"
              value={limit}
              onChange={(event) => {
                const val = Number(event.target.value);
                setLimit(Number.isFinite(val) && val > 0 ? val : DEFAULT_LIMIT);
              }}
              helperText="How many tickers to score"
              sx={{ minWidth: 160 }}
            />
            <TextField
              type="number"
              size="small"
              label="Strike count"
              value={strikeCount}
              onChange={(event) => {
                const val = Number(event.target.value);
                setStrikeCount(Number.isFinite(val) && val > 0 ? val : DEFAULT_STRIKE_COUNT);
              }}
              helperText="Strikes above/below ATM"
              sx={{ minWidth: 160 }}
            />
            <TextField
              select
              size="small"
              label="Contracts"
              value={includePuts ? "calls_puts" : "calls_only"}
              onChange={(event) => setIncludePuts(event.target.value === "calls_puts")}
              helperText="Include puts in the table"
              sx={{ minWidth: 160 }}
            >
              <MenuItem value="calls_puts">Calls + puts</MenuItem>
              <MenuItem value="calls_only">Calls only</MenuItem>
            </TextField>
          </Stack>
        </Stack>
        {historyError && <Alert severity="error">{historyError}</Alert>}
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={1}
          alignItems={{ xs: "flex-start", md: "center" }}
        >
          <HistoryCalendarDropdown
            label="Historical runs"
            value={historySelected}
            onChange={setHistorySelected}
            helperText={historyHelperText}
            historyRuns={historyRuns}
            disabled={historyLoading}
            minWidth={220}
          />
          <Button
            variant="outlined"
            startIcon={historyRunLoading ? <CircularProgress size={18} /> : <History />}
            onClick={loadHistoryRun}
            disabled={historyRunLoading || historySelected === HISTORY_LATEST}
          >
            {historyRunLoading ? "Loading..." : "Load history"}
          </Button>
          <Button
            variant="text"
            onClick={() => {
              setHistorySelected(HISTORY_LATEST);
              setPerformanceHistorySelected(HISTORY_LATEST);
            }}
            disabled={historyRunLoading}
          >
            View latest
          </Button>
          {viewLabel ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
              Viewing: {viewLabel}
            </Typography>
          ) : null}
        </Stack>
        <OptionsTable rows={optionsRows} includePuts={includePuts} />
      </Paper>

      <OptionsPerformanceTable
        rows={performanceRows}
        loading={performanceLoadingCombined}
        error={combinedPerformanceError}
        viewLabel={performanceViewLabel}
        lastRun={performanceLastRun}
        controls={
          <Stack
            direction={{ xs: "column", sm: "row" }}
            alignItems={{ xs: "flex-start", sm: "center" }}
            spacing={1}
          >
            <TextField
              select
              size="small"
              label="Performance run"
              value={performanceHistoryValue}
              onChange={(event) => {
                setPerformanceHistorySelected(event.target.value);
                setPerformanceRunError("");
              }}
              helperText={performanceHistoryHelperText}
              sx={{ minWidth: 220 }}
            >
              <MenuItem value={HISTORY_LATEST}>Latest (cached)</MenuItem>
              {historyRuns.map((entry) => (
                <MenuItem key={entry.date || entry.label} value={entry.date}>
                  {formatHistoryLabel(entry)}
                </MenuItem>
              ))}
            </TextField>
            {performanceHistoryValue !== HISTORY_LATEST ? (
              <Button
                variant="text"
                onClick={() => setPerformanceHistorySelected(HISTORY_LATEST)}
                disabled={performanceRunLoading}
              >
                View latest
              </Button>
            ) : null}
          </Stack>
        }
      />
    </Container>
  );
}
