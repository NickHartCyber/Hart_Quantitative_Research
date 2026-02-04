import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { History } from "@mui/icons-material";
import { Link as RouterLink } from "react-router-dom";
import HistoryCalendarDropdown from "../components/HistoryCalendarDropdown";

const API_BASE = process.env.REACT_APP_API_BASE || "";
const HISTORY_LATEST = "latest";
const DEFAULT_STOCK_LIST_ID = "mass_combo";
const MASS_COMBO_LABEL = "Mass combo list";

const SUGGESTER_CHOICES = [
  {
    key: "short_term",
    label: "1-2 day momentum",
    description: "Breakout-oriented entries sized for a quick 1-2 day hold.",
    defaultPeriod: "1y",
  },
  {
    key: "swing_term",
    label: "3-4 month swing",
    description: "Medium-term trend and 3-6 month momentum with wider stops/targets.",
    defaultPeriod: "3y",
  },
  {
    key: "long_term",
    label: "1+ year compounder",
    description: "Long-term trend, positive 12m returns, and lower realized volatility bias.",
    defaultPeriod: "5y",
  },
];

function formatHistoryLabel(entry) {
  const label = entry?.label || entry?.date || "";
  const count = Number(entry?.count);
  if (!label) {
    return "Unknown run";
  }
  if (Number.isFinite(count)) {
    return `${label} (${count} idea${count === 1 ? "" : "s"})`;
  }
  return label;
}

function formatHistoryViewLabel(label, count) {
  if (!label) {
    return "History run";
  }
  const num = Number(count);
  if (Number.isFinite(num)) {
    return `History: ${label} (${num} idea${num === 1 ? "" : "s"})`;
  }
  return `History: ${label}`;
}

function formatHistoryDateLabel(dateStr) {
  if (typeof dateStr !== "string") {
    return "";
  }
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

function formatCurrency(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return `$${Number(value).toFixed(2)}`;
}

function formatCell(value) {
  if (value === undefined || value === null || value === "") {
    return "--";
  }
  if (typeof value === "number") {
    const abs = Math.abs(value);
    const maximumFractionDigits = abs >= 100 ? 2 : abs >= 1 ? 3 : 4;
    return value.toLocaleString(undefined, { maximumFractionDigits: maximumFractionDigits });
  }
  return String(value);
}

function formatPercentChange(value) {
  if (!Number.isFinite(Number(value))) {
    return "--";
  }
  const num = Number(value);
  return `${num >= 0 ? "+" : ""}${num.toFixed(2)}%`;
}

function formatSignedCurrencyChange(value) {
  if (!Number.isFinite(Number(value))) {
    return "--";
  }
  const abs = Math.abs(Number(value));
  const base = formatCurrency(abs);
  return `${value >= 0 ? "+" : "-"}${base}`;
}

function extractTicker(row) {
  return String(row?.ticker || row?.symbol || row?.asset || "").trim().toUpperCase();
}

function pickRecommendationPrice(row) {
  const candidates = [row?.entry, row?.close, row?.price];
  for (const candidate of candidates) {
    const num = Number(candidate);
    if (Number.isFinite(num) && num !== 0) {
      return num;
    }
  }
  return null;
}

function pickQuoteCurrentPrice(quote) {
  if (!quote) {
    return null;
  }
  const mid = Number(quote.mid);
  if (Number.isFinite(mid)) {
    return mid;
  }
  const bid = Number(quote.bid);
  const ask = Number(quote.ask);
  if (Number.isFinite(bid) && Number.isFinite(ask)) {
    return (bid + ask) / 2;
  }
  const mark = Number(quote.mark);
  if (Number.isFinite(mark)) {
    return mark;
  }
  const last = Number(quote.last);
  if (Number.isFinite(last)) {
    return last;
  }
  const close = Number(quote.close);
  if (Number.isFinite(close)) {
    return close;
  }
  return null;
}

function tickerHref(value) {
  const trimmed = String(value || "").trim();
  if (!trimmed) {
    return "";
  }
  return `/ticker?ticker=${encodeURIComponent(trimmed)}`;
}

function renderTickerLink(value, { emphasize = false } = {}) {
  const trimmed = String(value || "").trim();
  if (!trimmed) {
    return "--";
  }
  return (
    <Link
      component={RouterLink}
      to={tickerHref(trimmed)}
      underline="hover"
      color="inherit"
      sx={emphasize ? { fontWeight: 700 } : undefined}
    >
      {trimmed}
    </Link>
  );
}

function createEmptyHorizonState() {
  return {
    suggestions: [],
    columns: [],
    lastRun: null,
    viewLabel: "Latest run",
    loading: false,
    error: "",
    historyRuns: [],
    historySelected: HISTORY_LATEST,
    historyLoading: false,
    historyError: "",
    historyRunLoading: false,
    performanceHistorySelected: HISTORY_LATEST,
    performanceViewLabel: "Latest run",
    performanceLastRun: null,
    performanceRunError: "",
    stockListId: "",
    stockListLabel: "",
  };
}

function DataFrameTable({ columns = [], rows = [] }) {
  const inferredColumns = useMemo(() => {
    if (Array.isArray(columns) && columns.length > 0) {
      return columns;
    }
    const discovered = [];
    rows.forEach((row) => {
      Object.keys(row || {}).forEach((key) => {
        if (!discovered.includes(key)) {
          discovered.push(key);
        }
      });
    });
    if (discovered.includes("ticker")) {
      return ["ticker", ...discovered.filter((c) => c !== "ticker")];
    }
    return discovered;
  }, [columns, rows]);

  if (!rows || rows.length === 0) {
    return <Alert severity="info">No suggestions yet. Check the scheduled run or load history.</Alert>;
  }

  return (
    <Box sx={{ overflowX: "auto" }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            {inferredColumns.map((col) => (
              <TableCell key={col}>{col}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row, idx) => (
            <TableRow key={row.ticker ? `${row.ticker}-${idx}` : idx}>
              {inferredColumns.map((col) => (
                <TableCell key={`${row.ticker || idx}-${col}`} align={typeof row[col] === "number" ? "right" : "left"}>
                  {["ticker", "symbol"].includes(String(col).toLowerCase())
                    ? renderTickerLink(row[col])
                    : formatCell(row[col])}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

function PerformanceTable({ rows, loading, error, viewLabel, lastRun, controls = null }) {
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
            Select a saved run and compare each recommendation to today&apos;s price.
          </Typography>
          {viewLabel && (
            <Typography variant="caption" color="text.secondary">
              Viewing: {viewLabel}
            </Typography>
          )}
          {lastRun && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
              Run timestamp: {lastRun}
            </Typography>
          )}
        </Box>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ xs: "flex-start", sm: "center" }}>
          {controls}
          {loading && <CircularProgress size={22} />}
        </Stack>
      </Stack>
      {error && <Alert severity="error">{error}</Alert>}
      {!loading && safeRows.length === 0 ? (
        <Alert severity="info">Select a run to see performance.</Alert>
      ) : (
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="right">% Change since Recommendation</TableCell>
                <TableCell align="right">$ Change per Share since Recommendation</TableCell>
                <TableCell>Ticker</TableCell>
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
                return (
                  <TableRow key={`${row.ticker || row.symbol || idx}-${idx}`}>
                    <TableCell align="right" sx={{ color: pctColor }}>
                      {formatPercentChange(row.percentChange)}
                    </TableCell>
                    <TableCell align="right" sx={{ color: dollarColor }}>
                      {formatSignedCurrencyChange(row.dollarChange)}
                    </TableCell>
                    <TableCell>{renderTickerLink(row.ticker, { emphasize: true })}</TableCell>
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
                <TableCell colSpan={6} sx={{ fontWeight: 700 }}>
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

export default function BuySellSuggesterPage() {
  const [horizonState, setHorizonState] = useState(() =>
    Object.fromEntries(SUGGESTER_CHOICES.map(({ key }) => [key, createEmptyHorizonState()])),
  );
  const [activeHorizon, setActiveHorizon] = useState(SUGGESTER_CHOICES[0].key);
  const [performanceSuggestions, setPerformanceSuggestions] = useState([]);
  const [performanceViewLabel, setPerformanceViewLabel] = useState("Latest run");
  const [performanceLastRun, setPerformanceLastRun] = useState(null);
  const [performanceRunLoading, setPerformanceRunLoading] = useState(false);
  const [performanceRunError, setPerformanceRunError] = useState("");
  const [priceLookup, setPriceLookup] = useState({});
  const [performanceError, setPerformanceError] = useState("");
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const runControllers = useRef({});
  const selectedStockList = DEFAULT_STOCK_LIST_ID;

  const activeState = horizonState[activeHorizon] || createEmptyHorizonState();

  const updateHorizonState = useCallback((key, updates) => {
    setHorizonState((prev) => {
      const prevState = prev[key] || createEmptyHorizonState();
      const nextState = typeof updates === "function" ? updates(prevState) : updates;
      return { ...prev, [key]: { ...prevState, ...nextState } };
    });
  }, []);

  const applyPayload = useCallback(
    (key, data, { viewLabel, stockList } = {}) => {
      const columns = Array.isArray(data?.columns) ? data.columns : [];
      const items = Array.isArray(data) ? data : data?.suggestions || data?.results || data?.items || [];
      const lastRunValue = data?.last_run || (items.length > 0 ? new Date().toISOString() : null);
      const listId = stockList?.id || data?.stock_list || "";
      const listLabel = stockList?.label || data?.stock_list_label || "";

      updateHorizonState(key, (prev) => ({
        ...prev,
        suggestions: items,
        columns,
        lastRun: lastRunValue || prev.lastRun,
        viewLabel: viewLabel || prev.viewLabel,
        historySelected: HISTORY_LATEST,
        performanceHistorySelected: HISTORY_LATEST,
        stockListId: listId || prev.stockListId,
        stockListLabel: listLabel || prev.stockListLabel,
        error: "",
        historyError: "",
      }));
    },
    [updateHorizonState],
  );

  const fetchHistoryRuns = useCallback(
    async (key) => {
      updateHorizonState(key, { historyLoading: true, historyError: "" });
      try {
        const res = await fetch(`${API_BASE}/api/suggest/buy-sell/history?horizon=${key}`);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const runs = Array.isArray(data) ? data : data.history || data.items || [];
        updateHorizonState(key, { historyRuns: runs });
      } catch (e) {
        updateHorizonState(key, { historyError: e.message || String(e) });
      } finally {
        updateHorizonState(key, { historyLoading: false });
      }
    },
    [updateHorizonState],
  );

  const runHorizonSuggester = useCallback(
    async (key, { force = false, background = false, cacheOnly = false } = {}) => {
      if (!background) {
        updateHorizonState(key, { loading: true, error: "" });
      }
      updateHorizonState(key, { historyError: "" });
      const controller = new AbortController();
      const currentControllers = runControllers.current || {};
      if (currentControllers[key]) {
        currentControllers[key].abort();
      }
      currentControllers[key] = controller;
      runControllers.current = currentControllers;
      let aborted = false;
      try {
        const payload = { horizon: key, use_cache: true };
        if (force) {
          payload.force = true;
        }
        if (cacheOnly) {
          payload.cache_only = true;
        }
        if (selectedStockList) {
          payload.stock_list = selectedStockList;
        }
        const res = await fetch(`${API_BASE}/api/suggest/buy-sell`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const viewLabel = "Latest run";
        const listPayload = selectedStockList
          ? { id: selectedStockList, label: MASS_COMBO_LABEL }
          : undefined;
        applyPayload(key, data, { viewLabel, stockList: listPayload });
      } catch (e) {
        if (e.name === "AbortError") {
          aborted = true;
          if (!background) {
            updateHorizonState(key, { error: "Run stopped." });
          }
          return;
        }
        if (!background) {
          updateHorizonState(key, { error: e.message || String(e) });
        }
      } finally {
        if (runControllers.current && runControllers.current[key] === controller) {
          delete runControllers.current[key];
        }
        if (!background) {
          updateHorizonState(key, { loading: false });
        }
        if (!aborted) {
          fetchHistoryRuns(key);
        }
      }
    },
    [applyPayload, fetchHistoryRuns, selectedStockList, updateHorizonState],
  );


  const loadHistoryRun = useCallback(
    async (key) => {
      const state = horizonState[key] || createEmptyHorizonState();
      if (!state.historySelected || state.historySelected === HISTORY_LATEST) {
        return;
      }
      updateHorizonState(key, { historyRunLoading: true, historyError: "", error: "" });
      try {
        const res = await fetch(
          `${API_BASE}/api/suggest/buy-sell/history/${state.historySelected}?horizon=${key}`,
        );
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const label = data?.label || formatHistoryDateLabel(state.historySelected);
        const count = data?.count ?? (Array.isArray(data?.suggestions) ? data.suggestions.length : 0);
        applyPayload(key, data, { viewLabel: formatHistoryViewLabel(label, count) });
      } catch (e) {
        updateHorizonState(key, { historyError: e.message || String(e) });
      } finally {
        updateHorizonState(key, { historyRunLoading: false });
      }
    },
    [applyPayload, horizonState, updateHorizonState],
  );

  const viewLatest = useCallback(
    (key) => {
      updateHorizonState(key, {
        historySelected: HISTORY_LATEST,
        performanceHistorySelected: HISTORY_LATEST,
      });
      runHorizonSuggester(key, { cacheOnly: true });
    },
    [runHorizonSuggester, updateHorizonState],
  );

  useEffect(() => {
    SUGGESTER_CHOICES.forEach(({ key }) => {
      fetchHistoryRuns(key);
      runHorizonSuggester(key, { background: true, cacheOnly: true });
    });
  }, [fetchHistoryRuns, runHorizonSuggester]);

  useEffect(() => {
    setPerformanceRunError("");
    setPerformanceError("");
  }, [activeHorizon]);

  useEffect(() => {
    if (activeState.performanceHistorySelected !== HISTORY_LATEST) {
      return;
    }
    setPerformanceSuggestions(activeState.suggestions);
    setPerformanceViewLabel(activeState.viewLabel);
    setPerformanceLastRun(activeState.lastRun);
    setPerformanceRunError("");
    setPerformanceRunLoading(false);
  }, [activeHorizon, activeState.performanceHistorySelected, activeState.suggestions, activeState.viewLabel, activeState.lastRun]);

  useEffect(() => {
    if (!activeState.performanceHistorySelected || activeState.performanceHistorySelected === HISTORY_LATEST) {
      return undefined;
    }

    let cancelled = false;
    setPerformanceRunLoading(true);
    setPerformanceRunError("");
    setPerformanceViewLabel(formatHistoryViewLabel(formatHistoryDateLabel(activeState.performanceHistorySelected)));
    setPerformanceLastRun(null);

    async function loadPerformanceHistory() {
      try {
        const res = await fetch(
          `${API_BASE}/api/suggest/buy-sell/history/${activeState.performanceHistorySelected}?horizon=${activeHorizon}`,
        );
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const items = Array.isArray(data)
          ? data
          : data.suggestions || data.results || data.items || [];
        const label = data?.label || formatHistoryDateLabel(activeState.performanceHistorySelected);
        const count = data?.count ?? (Array.isArray(items) ? items.length : 0);
        if (cancelled) return;
        setPerformanceSuggestions(items);
        setPerformanceViewLabel(formatHistoryViewLabel(label, count));
        setPerformanceLastRun(data?.last_run || data?.date || null);
      } catch (e) {
        if (!cancelled) {
          setPerformanceRunError(e.message || String(e));
          setPerformanceSuggestions([]);
          setPerformanceViewLabel(formatHistoryViewLabel(formatHistoryDateLabel(activeState.performanceHistorySelected)));
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
  }, [activeHorizon, activeState.performanceHistorySelected]);

  useEffect(() => {
    const tickers = Array.from(
      new Set(
        (performanceSuggestions || [])
          .map((row) => extractTicker(row))
          .filter((t) => t),
      ),
    );

    if (tickers.length === 0) {
      setPriceLookup({});
      setPerformanceError("");
      setPerformanceLoading(false);
      return;
    }

    let cancelled = false;
    setPerformanceLoading(true);
    setPerformanceError("");

    async function loadQuotes() {
      try {
        const res = await fetch(
          `${API_BASE}/api/quotes?symbols=${encodeURIComponent(tickers.join(","))}`,
        );
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const quotes = Array.isArray(data?.quotes) ? data.quotes : [];
        const missing = Array.isArray(data?.missing) ? data.missing : [];
        const next = {};
        const errors = [];

        quotes.forEach((quote) => {
          const symbol = String(quote?.symbol || "").trim().toUpperCase();
          if (!symbol) {
            return;
          }
          const price = pickQuoteCurrentPrice(quote);
          if (Number.isFinite(price)) {
            next[symbol] = price;
          } else {
            errors.push(`${symbol}: No quote price returned`);
          }
        });

        if (missing.length) {
          errors.push(`Missing quotes for ${missing.join(", ")}`);
        }
        if (cancelled) return;
        setPriceLookup(next);
        if (errors.length) {
          setPerformanceError(errors.join("; "));
        }
      } catch (e) {
        if (!cancelled) {
          setPerformanceError(e.message || String(e));
          setPriceLookup({});
        }
      } finally {
        if (!cancelled) {
          setPerformanceLoading(false);
        }
      }
    }

    loadQuotes();

    return () => {
      cancelled = true;
    };
  }, [performanceSuggestions]);

  const performanceRows = useMemo(
    () =>
      (performanceSuggestions || []).map((row) => {
        const ticker = extractTicker(row);
        const recommendationPrice = pickRecommendationPrice(row);
        const currentPrice = priceLookup[ticker];
        const dollarChange =
          Number.isFinite(recommendationPrice) && Number.isFinite(currentPrice)
            ? currentPrice - recommendationPrice
            : null;
        const percentChange =
          Number.isFinite(dollarChange) && Number.isFinite(recommendationPrice) && recommendationPrice !== 0
            ? (dollarChange / recommendationPrice) * 100
            : null;

        return {
          ...row,
          ticker,
          recommendationPrice,
          currentPrice,
          dollarChange,
          percentChange,
        };
      }),
    [performanceSuggestions, priceLookup],
  );

  const activeHistoryRuns = activeState.historyRuns || [];
  const oldestHistoryDate = useMemo(() => getOldestHistoryDate(activeHistoryRuns), [activeHistoryRuns]);
  const historyHelperText = activeState.historyLoading
    ? "Loading saved runs..."
    : activeHistoryRuns.length > 0
    ? `${activeHistoryRuns.length} saved run${activeHistoryRuns.length === 1 ? "" : "s"}${oldestHistoryDate ? ` | Oldest: ${oldestHistoryDate}` : ""}`
    : "No saved runs yet";
  const performanceHistoryHelperText =
    activeState.performanceHistorySelected === HISTORY_LATEST
      ? "Compare performance using the latest loaded suggestions"
      : `Performance for ${formatHistoryDateLabel(activeState.performanceHistorySelected)}`;
  const combinedPerformanceError = [performanceRunError, performanceError].filter(Boolean).join(" | ");
  const performanceLoadingCombined = performanceLoading || performanceRunLoading;
  const activeMeta = SUGGESTER_CHOICES.find((item) => item.key === activeHorizon) || SUGGESTER_CHOICES[0];
  const historySelected = activeState.historySelected || HISTORY_LATEST;
  const performanceHistorySelected = activeState.performanceHistorySelected || HISTORY_LATEST;

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 2.5 }}>
      <Stack
        direction={{ xs: "column", md: "row" }}
        alignItems={{ xs: "flex-start", md: "center" }}
        justifyContent="space-between"
        spacing={1.5}
      >
        <Box>
          <Typography variant="h5">Stock Ideas</Typography>
          <Typography variant="body2" color="text.secondary">
            Choose a horizon-specific script (fast 1-2 day, 3-4 month swing, or 1+ year hold), refreshed automatically at 7:30am on weekdays. Uses the {MASS_COMBO_LABEL}.
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
          {activeState.lastRun && (
            <Typography variant="caption" color="text.secondary">
              Last {activeMeta.label} run: {activeState.lastRun}
            </Typography>
          )}
        </Box>
      </Stack>
      {activeState.error && <Alert severity="error">{activeState.error}</Alert>}

      <Paper variant="outlined" sx={{ p: 2.5, display: "grid", gap: 1.5 }}>
        <Stack
          direction={{ xs: "column", md: "row" }}
          alignItems={{ xs: "flex-start", md: "center" }}
          justifyContent="space-between"
          spacing={1}
        >
          <Box>
            <Typography variant="h6">{activeMeta.label}</Typography>
            <Typography variant="body2" color="text.secondary">
              {activeMeta.description}
            </Typography>
            {activeState.lastRun && (
              <Typography variant="caption" color="text.secondary">
                Last run: {activeState.lastRun}
              </Typography>
            )}
          </Box>
        </Stack>
        {activeState.historyError && <Alert severity="error">{activeState.historyError}</Alert>}
        <Stack
          direction={{ xs: "column", md: "row" }}
          alignItems={{ xs: "flex-start", md: "center" }}
          spacing={1}
        >
          <HistoryCalendarDropdown
            label="Historical runs"
            value={historySelected}
            onChange={(next) => updateHorizonState(activeHorizon, { historySelected: next, historyError: "" })}
            helperText={historyHelperText}
            historyRuns={activeHistoryRuns}
            disabled={activeState.historyLoading}
            minWidth={240}
          />
          <Button
            variant="outlined"
            startIcon={activeState.historyRunLoading ? <CircularProgress size={18} /> : <History />}
            onClick={() => loadHistoryRun(activeHorizon)}
            disabled={
              activeState.historyRunLoading
              || activeState.historyLoading
              || historySelected === HISTORY_LATEST
            }
          >
            {activeState.historyRunLoading ? "Loading..." : "Load history"}
          </Button>
          <Button
            variant="text"
            onClick={() => viewLatest(activeHorizon)}
            disabled={activeState.loading}
          >
            View latest
          </Button>
        </Stack>
        {activeState.viewLabel && (
          <Typography variant="caption" color="text.secondary">
            Viewing: {activeState.viewLabel}
          </Typography>
        )}
        <DataFrameTable columns={activeState.columns} rows={activeState.suggestions} />
      </Paper>

      <PerformanceTable
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
              value={performanceHistorySelected}
              onChange={(event) => {
                updateHorizonState(activeHorizon, { performanceHistorySelected: event.target.value });
                setPerformanceRunError("");
              }}
              helperText={performanceHistoryHelperText}
              sx={{ minWidth: 240 }}
            >
              <MenuItem value={HISTORY_LATEST}>Latest (cached)</MenuItem>
              {activeHistoryRuns.map((entry) => (
                <MenuItem key={entry.date || entry.label} value={entry.date}>
                  {formatHistoryLabel(entry)}
                </MenuItem>
              ))}
            </TextField>
            {performanceHistorySelected !== HISTORY_LATEST && (
              <Button
                variant="text"
                onClick={() => updateHorizonState(activeHorizon, { performanceHistorySelected: HISTORY_LATEST })}
                disabled={performanceRunLoading}
              >
                View latest
              </Button>
            )}
          </Stack>
        }
      />
    </Container>
  );
}
