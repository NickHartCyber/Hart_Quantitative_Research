import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Container,
  Typography,
  Paper,
  TextField,
  MenuItem,
  Button,
  CircularProgress,
  Alert,
  Stack,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// reuse your global API_BASE
const API_BASE = process.env.REACT_APP_API_BASE || "";
const CACHE_KEY = "mg_dashboard_trades_v1";
const CACHE_DAYS_KEY = "mg_dashboard_selected_days";
const FETCH_TIMEOUT_MS = 20 * 1000;

function toArrayMaybe(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") return Object.values(value);
  return [];
}

// helper to format date labels a bit
function fmtDateLabel(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function formatTradeTime(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}

function formatNumber(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "";
  return num.toFixed(digits);
}

function formatCurrency(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return `$${num.toFixed(digits)}`;
}

// keep a tiny cache in localStorage so revisiting the page is instant
function readCachedTrades(days) {
  try {
    if (typeof window === "undefined") return null;
    const raw = window.localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const entry = parsed?.[String(days)];
    if (!entry || !Array.isArray(entry.trades)) return null;
    // equityCurve was added later; tolerate missing or malformed cache values
    if (entry.equityCurve && !Array.isArray(entry.equityCurve)) {
      entry.equityCurve = [];
    }
    return entry;
  } catch {
    return null;
  }
}

function writeCachedTrades(days, trades, equityCurve = [], fetchedAt = Date.now()) {
  try {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem(CACHE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    parsed[String(days)] = { trades, equityCurve, fetchedAt };
    window.localStorage.setItem(CACHE_KEY, JSON.stringify(parsed));
  } catch {
    // ignore storage failures (private mode, etc.)
  }
}

function getInitialDays() {
  try {
    if (typeof window === "undefined") return 30;
    const stored = Number.parseInt(window.localStorage.getItem(CACHE_DAYS_KEY) || "", 10);
    return Number.isFinite(stored) ? stored : 30;
  } catch {
    return 30;
  }
}

function formatLastUpdated(ts) {
  if (!ts) return "Not fetched yet";
  const d = new Date(ts);
  return `Last updated ${d.toLocaleString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    month: "short",
    day: "numeric",
  })}`;
}

export default function DashboardPage() {
  const [days, setDays] = useState(getInitialDays);
  const [trades, setTrades] = useState([]);
  const [equityCurve, setEquityCurve] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [source, setSource] = useState("");
  const mountedRef = useRef(true);
  const fetchIdRef = useRef(0);
  const pendingRef = useRef(0);

  // remember last mount to avoid setting state on unmounted component
  useEffect(() => {
    // In React StrictMode effects run twice; reassert mounted = true on each run.
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // persist the user's "days" selection
  useEffect(() => {
    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(CACHE_DAYS_KEY, String(days));
      }
    } catch {
      // ignore storage failures
    }
  }, [days]);

  const fetchTrades = useCallback(
    async ({ forceRefresh = false } = {}) => {
      const fetchId = ++fetchIdRef.current;
      const cached = readCachedTrades(days);

      if (cached) {
        setTrades(cached.trades);
        setEquityCurve(cached.equityCurve || []);
        setLastUpdated(cached.fetchedAt);
        setError("");
      }

      pendingRef.current += 1;
      setLoading(true);
      setError("");

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
        let res;
        try {
          res = await fetch(`${API_BASE}/api/trades?days=${days}&refresh=false`, { signal: controller.signal });
        } finally {
          clearTimeout(timeoutId);
        }
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || `HTTP ${res.status}`);
        }
        const data = await res.json();
        if (!mountedRef.current || fetchIdRef.current !== fetchId) return;
        const nextTrades = toArrayMaybe(
          Array.isArray(data) ? data : data?.trades || data?.trades_df || data?.tradesDf
        );
        const rawCurve = toArrayMaybe(data?.equity_curve || data?.equityCurve || data?.equitycurve);
        const dailyPnl = toArrayMaybe(data?.daily_pnl || data?.dailyPnl || data?.dailyPNL);

        // Normalize curve: prefer API equity_curve, then daily_pnl, then build from trades.
        let nextCurve = rawCurve;
        if (!nextCurve.length && dailyPnl.length) {
          let running = 0;
          nextCurve = dailyPnl
            .map((row) => {
              const pnl = Number(row.pnl ?? row.value ?? row.amount ?? 0);
              running += pnl;
              const time =
                row.date || row.day || row.time || row.timestamp || row.ts || new Date().toISOString();
              return { exit_time: time, pnl, cum_pnl: running };
            })
            .sort(
              (a, b) =>
                new Date(a.exit_time || a.time || a.date || 0).getTime() -
                new Date(b.exit_time || b.time || b.date || 0).getTime()
            );
        }
        if (!nextCurve.length && nextTrades.length) {
          let running = 0;
          nextCurve = [...nextTrades]
            .sort(
              (a, b) =>
                new Date(a.exit_time || a.entry_time || a.time || a.date || 0).getTime() -
                new Date(b.exit_time || b.entry_time || b.time || b.date || 0).getTime()
            )
            .map((t) => {
              const pnl = Number(t.pnl ?? t.realized_pnl ?? 0);
              running += pnl;
              return {
                exit_time: t.exit_time || t.entry_time || t.time || t.date || new Date().toISOString(),
                pnl,
                cum_pnl: running,
              };
            });
        }

        setTrades(nextTrades);
        setEquityCurve(nextCurve);
        const fetchedAt =
          data && typeof data === "object" && data.fetched_at
            ? new Date(data.fetched_at).getTime()
            : Date.now();
        setLastUpdated(fetchedAt);
        const nextSource = typeof data === "object" && data?.source ? data.source : "";
        setSource(nextSource);
        if (data?.live_error) {
          const cacheNote = nextSource.startsWith("cache:") ? " Showing cached data." : "";
          setError(`Live pull failed: ${data.live_error}.${cacheNote}`);
        } else {
          setError("");
        }
        writeCachedTrades(days, nextTrades, nextCurve, fetchedAt);
      } catch (e) {
        if (!mountedRef.current || fetchIdRef.current !== fetchId) return;
        const message =
          e?.name === "AbortError"
            ? `Request timed out after ${Math.round(FETCH_TIMEOUT_MS / 1000)}s`
            : e.message || String(e);
        setError(cached ? `Refresh failed: ${message} (showing cached data)` : message);
        if (!cached) {
          setTrades([]);
          setEquityCurve([]);
          setLastUpdated(null);
          setSource("");
        }
      } finally {
        pendingRef.current = Math.max(0, pendingRef.current - 1);
        if (mountedRef.current) setLoading(pendingRef.current > 0);
      }
    },
    [days]
  );

  // fetch trades when days changes (with a short-lived cache)
  useEffect(() => {
    fetchTrades();
  }, [days, fetchTrades]);

  // compute chart data + metrics
  const { chartData, totalPnL, winPct, lossPct, evenPct, totalTrades } = useMemo(() => {
    const curve = Array.isArray(equityCurve) ? equityCurve : [];
    const tradeList = Array.isArray(trades) ? trades : [];
    const chartSource = curve.length > 0 ? curve : tradeList;

    if (!chartSource.length) {
      return {
        chartData: [],
        totalPnL: 0,
        winPct: 0,
        lossPct: 0,
        evenPct: 0,
        totalTrades: 0,
      };
    }

    const sorted = [...chartSource].sort((a, b) => {
      const da = new Date(a.exit_time || a.entry_time || a.time || a.date || 0).getTime();
      const db = new Date(b.exit_time || b.entry_time || b.time || b.date || 0).getTime();
      return da - db;
    });

    // Build chart points. Prefer the precomputed cum_pnl if we have the equity curve.
    let running = 0;
    const points = sorted.map((row) => {
      const basePnl = Number(row.pnl ?? row.realized_pnl ?? 0);
      if (curve.length > 0) {
        const cum = Number.isFinite(Number(row.cum_pnl)) ? Number(row.cum_pnl) : (running += basePnl);
        running = cum;
        return {
          time: row.exit_time || row.time || row.date || new Date().toISOString(),
          pnl: cum,
        };
      }
      running += basePnl;
      return {
        time: row.exit_time || row.entry_time || row.date || new Date().toISOString(),
        pnl: running,
      };
    });

    // metrics
    const total = tradeList.length;
    let wins = 0,
      losses = 0,
      evens = 0,
      sum = 0;
    for (const row of tradeList) {
      const pnl = Number(row.pnl ?? row.realized_pnl ?? 0);
      sum += pnl;
      if (pnl > 0) wins += 1;
      else if (pnl < 0) losses += 1;
      else evens += 1;
    }

    return {
      chartData: points,
      totalPnL: curve.length > 0 ? points[points.length - 1]?.pnl ?? sum : sum,
      winPct: total ? (wins / total) * 100 : 0,
      lossPct: total ? (losses / total) * 100 : 0,
      evenPct: total ? (evens / total) * 100 : 0,
      totalTrades: total,
    };
  }, [trades, equityCurve]);

  const displayTrades = useMemo(() => {
    const tradeList = Array.isArray(trades) ? trades : [];
    return [...tradeList].sort((a, b) => {
      const da = new Date(a.exit_time || a.entry_time || a.time || a.date || 0).getTime();
      const db = new Date(b.exit_time || b.entry_time || b.time || b.date || 0).getTime();
      return db - da;
    });
  }, [trades]);

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "flex", flexDirection: "column", gap: 3 }}>
      {/* Header + controls */}
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        alignItems={{ xs: "flex-start", sm: "center" }}
        justifyContent="space-between"
      >
        <Box>
          <Typography variant="h5" gutterBottom>
            Nick's Trades
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Cumulative PnL for Nick's trades over the last {days} day(s).
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {formatLastUpdated(lastUpdated)} &bull; Uses saved snapshot
          </Typography>
          {source && (
            <Typography variant="caption" color="text.secondary" display="block">
              Source: {source.startsWith("cache:") ? `Cached (${source.slice(6)})` : "Live Schwab pull"}
            </Typography>
          )}
          {error && (trades.length > 0 || equityCurve.length > 0) && (
            <Typography variant="caption" color="error.main" display="block">
              {error}
            </Typography>
          )}
        </Box>
        <Stack direction="row" spacing={1} alignItems="center">
          <TextField
            select
            size="small"
            label="Days"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            sx={{ minWidth: 140 }}
          >
            <MenuItem value={7}>Last 7 days</MenuItem>
            <MenuItem value={30}>Last 30 days</MenuItem>
            <MenuItem value={60}>Last 60 days</MenuItem>
            <MenuItem value={90}>Last 90 days</MenuItem>
          </TextField>
          <Button
            variant="outlined"
            size="small"
            onClick={() => fetchTrades({ forceRefresh: true })}
            disabled={loading}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
          {loading && trades.length > 0 && <CircularProgress size={18} />}
        </Stack>
      </Stack>

      {/* Chart card */}
      <Paper sx={{ p: 2, height: 360, display: "flex", flexDirection: "column", gap: 2 }}>
        <Typography variant="subtitle1">Cumulative PnL</Typography>
        <Divider />
        <Box sx={{ flex: 1, minHeight: 240 }}>
          {loading && chartData.length === 0 ? (
            <Box sx={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <CircularProgress />
            </Box>
          ) : error && chartData.length === 0 ? (
            <Alert severity="error">{error}</Alert>
          ) : chartData.length === 0 ? (
            <Box sx={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Typography color="text.secondary">No trades in this period.</Typography>
            </Box>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                    dataKey="time"
                    tickFormatter={fmtDateLabel}
                    minTickGap={20}
                />
                <YAxis />
                <Tooltip
                  labelFormatter={(v) => new Date(v).toLocaleString()}
                  formatter={(value) => [`$${Number(value).toFixed(2)}`, "Cumulative PnL"]}
                />
                <Line
                  type="monotone"
                  dataKey="pnl"
                  stroke="#1976d2"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Box>
      </Paper>

      {/* Metrics row */}
      <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="overline">Total PnL</Typography>
          <Typography variant="h5" color={totalPnL >= 0 ? "success.main" : "error.main"}>
            ${totalPnL.toFixed(2)}
          </Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="overline">Trades</Typography>
          <Typography variant="h5">{totalTrades}</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="overline">Win %</Typography>
          <Typography variant="h5">{winPct.toFixed(1)}%</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="overline">Loss %</Typography>
          <Typography variant="h5">{lossPct.toFixed(1)}%</Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1 }}>
          <Typography variant="overline">Even %</Typography>
          <Typography variant="h5">{evenPct.toFixed(1)}%</Typography>
        </Paper>
      </Stack>

      {/* Trades table */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Trade Details
        </Typography>
        <Divider sx={{ mb: 2 }} />
        {displayTrades.length === 0 ? (
          <Typography color="text.secondary">No trades available.</Typography>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Date</TableCell>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Direction</TableCell>
                  <TableCell align="right">Qty</TableCell>
                  <TableCell align="right">Entry</TableCell>
                  <TableCell align="right">Exit</TableCell>
                  <TableCell align="right">PnL</TableCell>
                  <TableCell align="right">Return %</TableCell>
                  <TableCell align="right">Hold (min)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {displayTrades.map((trade, idx) => {
                  const pnl = Number(trade.pnl ?? trade.realized_pnl ?? 0);
                  const ret = Number(trade.return_pct ?? trade.returnPct ?? 0);
                  const retPct = Math.abs(ret) <= 1 ? ret * 100 : ret;
                  const retLabel = formatNumber(retPct);
                  return (
                    <TableRow key={`${trade.symbol || "trade"}-${trade.exit_time || trade.entry_time}-${idx}`}>
                      <TableCell>
                        {formatTradeTime(trade.exit_time || trade.entry_time || trade.time || trade.date)}
                      </TableCell>
                      <TableCell>{trade.symbol || "-"}</TableCell>
                      <TableCell>{trade.direction || trade.side || "-"}</TableCell>
                      <TableCell align="right">{formatNumber(trade.qty ?? trade.quantity ?? trade.shares, 0) || "-"}</TableCell>
                      <TableCell align="right">{formatCurrency(trade.entry_price ?? trade.entryPrice)}</TableCell>
                      <TableCell align="right">{formatCurrency(trade.exit_price ?? trade.exitPrice)}</TableCell>
                      <TableCell align="right" sx={{ color: pnl >= 0 ? "success.main" : "error.main" }}>
                        {formatCurrency(pnl)}
                      </TableCell>
                      <TableCell align="right">{retLabel ? `${retLabel}%` : "-"}</TableCell>
                      <TableCell align="right">{formatNumber(trade.holding_minutes ?? trade.holdingMinutes) || "-"}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>
    </Container>
  );
}
