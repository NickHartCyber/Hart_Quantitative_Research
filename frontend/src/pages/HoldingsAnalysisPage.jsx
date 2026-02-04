import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
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
import { Add, Delete, DeleteForever, Refresh, Replay, Restore, Save } from "@mui/icons-material";

const API_BASE = process.env.REACT_APP_API_BASE || "";
const HOLDINGS_FILE_LABEL = "files/logs/holdings.json";

const EMPTY_ROW = { ticker: "", shares: "", cost_basis: "" };
const SAMPLE_ROWS = [
  { ticker: "AAPL", shares: 10, cost_basis: 165 },
  { ticker: "MSFT", shares: 6, cost_basis: 280 },
  { ticker: "NVDA", shares: 4, cost_basis: 410 },
];

function formatCurrency(value) {
  if (!Number.isFinite(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(
    Number(value),
  );
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return "--";
  return `${Number(value).toFixed(2)}%`;
}

function formatNumber(value) {
  if (!Number.isFinite(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function RecommendationChip({ value }) {
  const normalized = String(value || "").toLowerCase();
  const config =
    normalized === "buy_more"
      ? { label: "Buy more", color: "success" }
      : normalized === "sell"
      ? { label: "Sell / Trim", color: "error" }
      : { label: "Hold", color: "info" };

  return <Chip label={config.label} color={config.color} size="small" />;
}

function normalizeRowsFromHoldings(rows = []) {
  if (!Array.isArray(rows)) return [{ ...EMPTY_ROW }];
  const normalized = rows.map((row) => ({
    ticker: String(row?.ticker || ""),
    shares: row?.shares ?? "",
    cost_basis: row?.cost_basis ?? "",
  }));
  return normalized.length ? normalized : [{ ...EMPTY_ROW }];
}

export default function HoldingsAnalysisPage() {
  const [rows, setRows] = useState(SAMPLE_ROWS);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState(null);
  const [saving, setSaving] = useState(false);
  const [loadingSaved, setLoadingSaved] = useState(false);
  const [clearing, setClearing] = useState(false);

  const cleanedHoldings = useMemo(() => {
    return rows
      .map((row) => {
        const ticker = String(row.ticker || "").trim().toUpperCase();
        const sharesNum = Number(row.shares);
        const basisNum = Number(row.cost_basis);
        return {
          ticker,
          shares: Number.isFinite(sharesNum) ? sharesNum : undefined,
          cost_basis: Number.isFinite(basisNum) ? basisNum : undefined,
        };
      })
      .filter((row) => row.ticker);
  }, [rows]);

  const summary = analysis?.summary || {};
  const holdings = analysis?.holdings || [];
  const asOfLabel = summary?.as_of ? new Date(summary.as_of).toLocaleString() : null;

  const setNoticeMessage = useCallback((text, severity = "success") => {
    setNotice(text ? { text, severity } : null);
  }, []);

  const fetchSavedHoldings = useCallback(async ({ silent = false } = {}) => {
    setLoadingSaved(true);
    if (!silent) {
      setNoticeMessage("");
    }
    try {
      const res = await fetch(`${API_BASE}/api/holdings/saved`);
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        const detail = payload?.detail || res.statusText || "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const data = await res.json();
      const savedHoldings = normalizeRowsFromHoldings(data?.holdings || []);
      const hasSavedTickers = savedHoldings.some((row) => row.ticker);
      if (hasSavedTickers) {
        setRows(savedHoldings);
        setAnalysis(null);
        const savedAtLabel = data?.saved_at ? new Date(data.saved_at).toLocaleString() : null;
        const pathLabel = data?.path || HOLDINGS_FILE_LABEL;
        setNoticeMessage(
          `Loaded saved holdings from ${pathLabel}${savedAtLabel ? ` (saved ${savedAtLabel})` : ""}.`,
          "success",
        );
      } else if (!silent) {
        setRows(savedHoldings);
        setAnalysis(null);
        setNoticeMessage("No saved holdings found yet. Add tickers and click Save holdings.", "info");
      }
    } catch (err) {
      if (!silent) {
        setNoticeMessage(err.message || "Couldn't load saved holdings.", "error");
      }
    } finally {
      setLoadingSaved(false);
    }
  }, [setNoticeMessage]);

  useEffect(() => {
    fetchSavedHoldings({ silent: true });
  }, [fetchSavedHoldings]);

  async function runAnalysis() {
    if (!cleanedHoldings.length) {
      setError("Add at least one holding with a ticker.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/holdings/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings: cleanedHoldings }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        const detail = payload?.detail || res.statusText || "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const data = await res.json();
      setAnalysis(data);
    } catch (err) {
      setError(err.message || "Failed to run analysis");
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }

  function updateRow(idx, field, value) {
    setNoticeMessage("");
    setRows((prev) => prev.map((row, i) => (i === idx ? { ...row, [field]: value } : row)));
  }

  function addRow() {
    setNoticeMessage("");
    setRows((prev) => [...prev, { ...EMPTY_ROW }]);
  }

  function removeRow(idx) {
    setNoticeMessage("");
    setRows((prev) => prev.filter((_, i) => i !== idx));
  }

  function resetToSample() {
    setRows(SAMPLE_ROWS);
    setAnalysis(null);
    setError("");
    setNoticeMessage("Loaded sample holdings. Click Save holdings to keep them on disk.");
  }

  async function saveHoldings() {
    setNoticeMessage("");
    setSaving(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/holdings/saved`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holdings: cleanedHoldings }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        const detail = payload?.detail || res.statusText || "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const data = await res.json();
      const savedAtLabel = data?.saved_at ? new Date(data.saved_at).toLocaleString() : "now";
      const pathLabel = data?.path || HOLDINGS_FILE_LABEL;
      setNoticeMessage(
        `Saved ${cleanedHoldings.length} holding${cleanedHoldings.length === 1 ? "" : "s"} to ${pathLabel} (saved ${savedAtLabel}).`,
        "success",
      );
    } catch (err) {
      setNoticeMessage(err.message || "Couldn't save holdings.", "error");
    } finally {
      setSaving(false);
    }
  }

  async function restoreSavedHoldings() {
    await fetchSavedHoldings({ silent: false });
  }

  async function clearSavedHoldings() {
    setNoticeMessage("");
    setError("");
    setClearing(true);
    try {
      const res = await fetch(`${API_BASE}/api/holdings/saved`, { method: "DELETE" });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        const detail = payload?.detail || res.statusText || "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      setRows([{ ...EMPTY_ROW }]);
      setAnalysis(null);
      setNoticeMessage("Cleared saved holdings file. Add rows to start fresh.", "info");
    } catch (err) {
      setNoticeMessage(err.message || "Couldn't clear saved holdings.", "error");
    } finally {
      setClearing(false);
    }
  }

  return (
    <Container maxWidth="lg">
      <Stack spacing={3}>
        <Box>
          <Typography variant="h4" fontWeight={800} gutterBottom>
            Holdings Checkup
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Drop in your current positions to see quick guidance on whether to hold, buy more, or sell.
            We look at P&L, short-term momentum, long-term trend, and 52-week levels for each ticker.
          </Typography>
        </Box>

        <Paper elevation={2} sx={{ p: 2.5 }}>
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems={{ xs: "flex-start", md: "center" }}
            justifyContent="space-between"
          >
            <Box>
              <Typography variant="h6" fontWeight={700}>
                Your holdings
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                Remove tickers or edit share counts, then hit Save to persist to {HOLDINGS_FILE_LABEL} on disk.
              </Typography>
              {notice ? (
                <Alert severity={notice.severity || "info"} sx={{ mt: 1 }}>
                  {notice.text}
                </Alert>
              ) : null}
            </Box>
            <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="flex-end" rowGap={1}>
              <Button
                startIcon={saving ? <CircularProgress size={18} color="inherit" /> : <Save />}
                onClick={saveHoldings}
                variant="contained"
                disabled={saving}
              >
                Save holdings
              </Button>
              <Button
                startIcon={loadingSaved ? <CircularProgress size={18} color="inherit" /> : <Restore />}
                onClick={restoreSavedHoldings}
                variant="outlined"
                disabled={loadingSaved}
              >
                Load saved
              </Button>
              <Button startIcon={<Refresh />} onClick={resetToSample} variant="outlined">
                Reset samples
              </Button>
              <Button
                startIcon={clearing ? <CircularProgress size={18} color="inherit" /> : <DeleteForever />}
                onClick={clearSavedHoldings}
                variant="outlined"
                color="error"
                disabled={clearing}
              >
                Clear saved
              </Button>
              <Button startIcon={<Add />} onClick={addRow} variant="outlined">
                Add row
              </Button>
            </Stack>
          </Stack>
          <Table size="small" sx={{ mt: 2 }}>
            <TableHead>
              <TableRow>
                <TableCell>Ticker</TableCell>
                <TableCell>Shares</TableCell>
                <TableCell>Cost basis ($)</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row, idx) => (
                <TableRow key={`holding-${idx}`}>
                  <TableCell width="20%">
                    <TextField
                      fullWidth
                      value={row.ticker}
                      onChange={(e) => updateRow(idx, "ticker", e.target.value.toUpperCase())}
                      placeholder="AAPL"
                      size="small"
                    />
                  </TableCell>
                  <TableCell width="25%">
                    <TextField
                      fullWidth
                      value={row.shares}
                      onChange={(e) => updateRow(idx, "shares", e.target.value)}
                      placeholder="10"
                      size="small"
                      type="number"
                      inputProps={{ min: "0", step: "0.01" }}
                    />
                  </TableCell>
                  <TableCell width="25%">
                    <TextField
                      fullWidth
                      value={row.cost_basis}
                      onChange={(e) => updateRow(idx, "cost_basis", e.target.value)}
                      placeholder="150"
                      size="small"
                      type="number"
                      inputProps={{ min: "0", step: "0.01" }}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <Button
                      aria-label="Remove stock from holdings"
                      onClick={() => removeRow(idx)}
                      size="small"
                      color="error"
                      startIcon={<Delete />}
                    >
                      Remove
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} mt={2} alignItems="center">
            <Button
              variant="contained"
              color="primary"
              onClick={runAnalysis}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={18} color="inherit" /> : <Replay />}
            >
              {loading ? "Crunching..." : "Analyze holdings"}
            </Button>
            <Typography variant="body2" color="text.secondary">
              Shares and cost basis are optional; we still score trend and momentum when they are missing.
            </Typography>
          </Stack>
          {error ? (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          ) : null}
        </Paper>

        <Paper elevation={2} sx={{ p: 2.5 }}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ xs: "flex-start", sm: "center" }}>
            <Typography variant="h6" fontWeight={700}>
              Results
            </Typography>
            {asOfLabel ? (
              <Chip label={`As of ${asOfLabel}`} size="small" sx={{ fontWeight: 600 }} />
            ) : null}
          </Stack>

          {!analysis && (
            <Alert severity="info" sx={{ mt: 2 }}>
              Run the analysis to see position-by-position recommendations.
            </Alert>
          )}

          {analysis ? (
            <Box mt={2}>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} mb={2}>
                <SummaryStat label="Portfolio value" value={formatCurrency(summary.total_value)} />
                <SummaryStat label="Cost basis" value={formatCurrency(summary.total_cost)} />
                <SummaryStat
                  label="Unrealized P&L"
                  value={`${formatCurrency(summary.total_unrealized_pl)} (${formatPercent(
                    summary.total_unrealized_pl_pct,
                  )})`}
                />
              </Stack>

              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Ticker</TableCell>
                    <TableCell>Recommendation</TableCell>
                    <TableCell>Price</TableCell>
                    <TableCell>Cost basis</TableCell>
                    <TableCell>Shares</TableCell>
                    <TableCell>Value</TableCell>
                    <TableCell>P&L</TableCell>
                    <TableCell>1m return</TableCell>
                    <TableCell>3m return</TableCell>
                    <TableCell>50d / 200d</TableCell>
                    <TableCell>52w spot</TableCell>
                    <TableCell>Notes</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {holdings.map((row, idx) => {
                    const key = row.ticker || row.name || `holding-${idx}`;
                    const plValue =
                      Number.isFinite(Number(row.unrealized_pl)) && Number.isFinite(Number(row.unrealized_pl_pct))
                        ? `${formatCurrency(row.unrealized_pl)} (${formatPercent(row.unrealized_pl_pct)})`
                        : formatCurrency(row.unrealized_pl) !== "--"
                          ? formatCurrency(row.unrealized_pl)
                          : "--";
                    const rangePos = Number(row.fifty_two_week_position);
                    const hasRange = Number.isFinite(rangePos);
                    const rangeSpot = hasRange ? formatPercent(rangePos * 100) : "--";
                    return (
                      <TableRow key={key}>
                        <TableCell>
                          <Typography fontWeight={700}>{row.ticker}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {row.name || ""}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <RecommendationChip value={row.recommendation} />
                        </TableCell>
                        <TableCell>{formatCurrency(row.current_price)}</TableCell>
                        <TableCell>{formatCurrency(row.cost_basis)}</TableCell>
                        <TableCell>{formatNumber(row.shares)}</TableCell>
                        <TableCell>{formatCurrency(row.market_value)}</TableCell>
                        <TableCell>{plValue}</TableCell>
                        <TableCell>{formatPercent(row.one_month_return)}</TableCell>
                        <TableCell>{formatPercent(row.three_month_return)}</TableCell>
                        <TableCell>
                          {formatCurrency(row.fifty_day_sma)} / {formatCurrency(row.two_hundred_day_sma)}
                        </TableCell>
                        <TableCell>{hasRange ? rangeSpot : "--"}</TableCell>
                        <TableCell>
                          <Typography variant="body2">{row.rationale || "—"}</Typography>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </Box>
          ) : null}
        </Paper>
      </Stack>
    </Container>
  );
}

function SummaryStat({ label, value }) {
  return (
    <Paper
      variant="outlined"
      sx={{
        flex: 1,
        p: 1.5,
        borderRadius: 2,
        minWidth: 200,
      }}
    >
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="h6" fontWeight={800}>
        {value || "--"}
      </Typography>
    </Paper>
  );
}
