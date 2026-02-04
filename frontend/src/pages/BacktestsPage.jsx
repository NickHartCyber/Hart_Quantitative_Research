import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  Divider,
  ButtonGroup,
  FormControlLabel,
  IconButton,
  MenuItem,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { History, InfoOutlined, PlayCircle, Refresh } from "@mui/icons-material";
import {
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = process.env.REACT_APP_API_BASE || "";

const fmtMoney = (val) => {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  const num = Number(val);
  const sign = num >= 0 ? "" : "-";
  return `${sign}$${Math.abs(num).toFixed(2)}`;
};

const fmtPct = (val) => {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return `${Number(val).toFixed(2)}%`;
};

const fmtNumber = (val) => {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return Number(val).toLocaleString();
};

const fmtDate = (iso) => {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch (e) {
    return iso;
  }
};

const fmtDay = (iso) => {
  if (!iso) return "—";
  const str = String(iso);
  if (/^\d{4}-\d{2}-\d{2}/.test(str)) {
    return str.slice(0, 10);
  }
  try {
    return new Date(str).toISOString().slice(0, 10);
  } catch (e) {
    return str;
  }
};

const sortPreviewRows = (rows = [], sortState) => {
  if (!sortState?.column) return rows;
  const { column, direction } = sortState;
  const factor = direction === "desc" ? -1 : 1;
  return [...rows].sort((a, b) => {
    const av = a?.[column];
    const bv = b?.[column];
    const aNum = Number(av);
    const bNum = Number(bv);
    const aIsNum = Number.isFinite(aNum);
    const bIsNum = Number.isFinite(bNum);
    if (aIsNum && bIsNum) {
      return (aNum - bNum) * factor;
    }
    const aStr = av == null ? "" : String(av);
    const bStr = bv == null ? "" : String(bv);
    if (aStr === bStr) return 0;
    return aStr > bStr ? factor : -factor;
  });
};

const fmtDuration = (seconds) => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const s = Math.max(0, Number(seconds));
  if (s < 60) return `${s.toFixed(1)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  return `${(s / 3600).toFixed(2)}h`;
};

const formatConfigLabel = (key) =>
  String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());

const CONFIG_DESCRIPTIONS = {
  ema_fast: "Fast EMA length for short-term trend.",
  ema_med: "Medium EMA in the trend stack.",
  ema_slow: "Slow EMA anchor for overall trend.",
  min_adx: "Minimum ADX(14) to accept a breakout.",
  thrust_atr_mult: "Signal bar range vs ATR gate.",
  min_body_frac: "Body fraction of bar range.",
  lookback_break: "Bars used to define breakout high.",
  entry_buffer_atr: "ATR buffer above breakout trigger.",
  initial_stop_atr: "Initial stop distance in ATR.",
  trail_atr_mult: "ATR multiple for trailing stop after move in your favor.",
  take_profit_R: "Take-profit at R multiples (R = entry−initial stop).",
  max_hold_minutes: "Time stop; closes any position after this many minutes.",
  rvol_recent: "Recent volume window for rVol.",
  rvol_window_baseline: "Baseline volume window (excludes recent window).",
  min_dollar_bar_reg: "Min dollar/bar (price×volume) during RTH.",
  min_dollar_bar_pre: "Min dollar/bar during premarket.",
  allow_premarket: "Allow signals and trading in premarket.",
  enable_cost_guards: "Master toggle for cost guards (price floor, ATR vs slippage, expected move vs cost, min TP).",
  min_price_dollars: "Skip names below this price.",
  atr_vs_slip_min_mult: "Require ATR ≥ this × per-side slippage.",
  exp_move_vs_cost_min_mult: "Require expected move ≥ this × round-trip slippage.",
  take_profit_min_cents: "Floor for take-profit in cents.",
  use_stop_limit: "Use stop-limit entry instead of stop-market.",
  stop_limit_offset_cents: "Limit cap above trigger for stop-limit.",
  slippage_assumed: "Assumed per-side slippage ($) for guards and logging.",
  slippage_per_share: "Per-side slippage used by tester P&L accounting.",
};

const normalizeConfigDraft = (config) => {
  const draft = {};
  const types = {};
  Object.entries(config || {}).forEach(([key, value]) => {
    if (typeof value === "number") {
      types[key] = "number";
      draft[key] = Number.isFinite(value) ? String(value) : "";
    } else if (typeof value === "boolean") {
      types[key] = "boolean";
      draft[key] = value;
    } else {
      types[key] = "string";
      draft[key] = value ?? "";
    }
  });
  return { draft, types };
};

const renderConfigLabel = (key) => {
  const desc = CONFIG_DESCRIPTIONS[key];
  return (
    <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5 }}>
      {formatConfigLabel(key)}
      {desc ? (
        <Tooltip
          title={desc}
          placement="top"
          enterTouchDelay={0}
          leaveTouchDelay={4000}
        >
          <IconButton
            size="small"
            edge="end"
            tabIndex={-1}
            onClick={(e) => e.stopPropagation()}
            aria-label={`${formatConfigLabel(key)} info`}
          >
            <InfoOutlined fontSize="inherit" />
          </IconButton>
        </Tooltip>
      ) : null}
    </Box>
  );
};

function SummaryTable({ records = [] }) {
  if (!records || records.length === 0) {
    return <Alert severity="info">No results yet. Run a backtest to see performance stats.</Alert>;
  }
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Config</TableCell>
          <TableCell align="right">Total P/L</TableCell>
          <TableCell align="right">Trades</TableCell>
          <TableCell align="right">Win %</TableCell>
          <TableCell align="right">Loss %</TableCell>
          <TableCell align="right">Profit Factor</TableCell>
          <TableCell align="right">Sharpe</TableCell>
          <TableCell align="right">Max DD</TableCell>
          <TableCell align="right">CAGR</TableCell>
          <TableCell align="right">Calmar</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {records.map((row) => (
          <TableRow key={row.config || row.config_name}>
            <TableCell>{row.config || row.config_name || "Config"}</TableCell>
            <TableCell align="right">{fmtMoney(row.total_pnl)}</TableCell>
            <TableCell align="right">{fmtNumber(row.num_trades)}</TableCell>
            <TableCell align="right">{fmtPct(row.win_rate_pct)}</TableCell>
            <TableCell align="right">{fmtPct(row.loss_rate_pct)}</TableCell>
            <TableCell align="right">{row.profit_factor != null ? Number(row.profit_factor).toFixed(2) : "—"}</TableCell>
            <TableCell align="right">{row.sharpe != null ? Number(row.sharpe).toFixed(2) : "—"}</TableCell>
            <TableCell align="right">{fmtPct(row.max_drawdown_pct)}</TableCell>
            <TableCell align="right">{fmtPct(row.cagr_pct)}</TableCell>
            <TableCell align="right">{row.calmar != null ? Number(row.calmar).toFixed(2) : "—"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function BacktestsPage() {
  const [configs, setConfigs] = useState([]);
  const [configId, setConfigId] = useState("");
  const [configBase, setConfigBase] = useState(null);
  const [configDraft, setConfigDraft] = useState({});
  const [configTypes, setConfigTypes] = useState({});
  const [pendingOverrides, setPendingOverrides] = useState(null);

  const [status, setStatus] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [starting, setStarting] = useState(false);
  const [slippageOverride, setSlippageOverride] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const [history, setHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runDetail, setRunDetail] = useState(null);
  const [loadingRunDetail, setLoadingRunDetail] = useState(false);
  const [tradesPage, setTradesPage] = useState(1);
  const [tradesLimit, setTradesLimit] = useState(25);
  const [tradeSort, setTradeSort] = useState({});

  const [error, setError] = useState("");

  const fetchConfigs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/backtests/configs`);
      if (!res.ok) {
        throw new Error(`Failed to load configs (${res.status})`);
      }
      const data = await res.json();
      const list = data.configs || [];
      setConfigs(list);
      if (!configId && list.length > 0) {
        const preferred = list.find((c) => c.id === "baseline")?.id || list[0].id;
        setConfigId(preferred);
      }
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [configId]);

  const fetchConfigDetail = useCallback(async (id) => {
    if (!id) return;
    try {
      const res = await fetch(`${API_BASE}/api/backtests/configs/${id}`);
      if (!res.ok) {
        throw new Error(`Failed to load config (${res.status})`);
      }
      const data = await res.json();
      const config = data?.config || {};
      const { draft, types } = normalizeConfigDraft(config);
      setConfigBase(config);
      setConfigDraft(draft);
      setConfigTypes(types);
    } catch (e) {
      setError(e.message || String(e));
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    setLoadingStatus(true);
    try {
      const res = await fetch(`${API_BASE}/api/backtests/status`);
      if (!res.ok) {
        throw new Error(`Failed to load status (${res.status})`);
      }
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE}/api/backtests/history`);
      if (!res.ok) {
        throw new Error(`Failed to load history (${res.status})`);
      }
      const data = await res.json();
      const runs = data?.runs || [];
      setHistory(runs);
      if (!selectedRunId && runs.length > 0) {
        setSelectedRunId(runs[0].id);
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoadingHistory(false);
    }
  }, [selectedRunId]);

  const fetchRunDetail = useCallback(async (runId) => {
    if (!runId) return;
    setLoadingRunDetail(true);
    try {
      const offset = Math.max(0, (tradesPage - 1) * tradesLimit);
      const res = await fetch(
        `${API_BASE}/api/backtests/history/${runId}?include_trades=true&trades_limit=${encodeURIComponent(
          tradesLimit || 25
        )}&trades_offset=${encodeURIComponent(offset)}`
      );
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Failed to load run ${runId}`);
      }
      const data = await res.json();
      setRunDetail(data?.run || null);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoadingRunDetail(false);
    }
  }, [tradesLimit, tradesPage]);

  const resetConfigDraft = useCallback(() => {
    if (!configBase) return;
    const { draft, types } = normalizeConfigDraft(configBase);
    setConfigDraft(draft);
    setConfigTypes(types);
  }, [configBase]);

  const buildConfigPayload = useCallback(() => {
    if (!configDraft || Object.keys(configDraft).length === 0) {
      return { payload: null, invalid: [] };
    }
    const payload = {};
    const invalid = [];
    Object.entries(configDraft).forEach(([key, value]) => {
      const type = configTypes[key];
      if (type === "number") {
        const raw = String(value ?? "");
        const num = Number(raw);
        if (!raw.trim() || Number.isNaN(num)) {
          invalid.push(key);
          return;
        }
        payload[key] = num;
        return;
      }
      if (type === "boolean") {
        payload[key] = Boolean(value);
        return;
      }
      if (value !== undefined) {
        payload[key] = value;
      }
    });
    return { payload, invalid };
  }, [configDraft, configTypes]);

  const handleConfigValueChange = useCallback((key, value) => {
    setConfigDraft((prev) => ({ ...prev, [key]: value }));
  }, []);

  const handleConfigToggle = useCallback((key, checked) => {
    setConfigDraft((prev) => ({ ...prev, [key]: checked }));
  }, []);

  const handleTradesLimitChange = useCallback((limit) => {
    setTradesLimit(limit);
    setTradesPage(1);
  }, []);

  const handleTradesPageChange = useCallback((page) => {
    setTradesPage((prev) => Math.max(1, page));
  }, []);

  const handleTradeSort = useCallback((cfg, column) => {
    setTradeSort((prev) => {
      const existing = prev[cfg] || {};
      const direction = existing.column === column && existing.direction === "asc" ? "desc" : "asc";
      return { ...prev, [cfg]: { column, direction } };
    });
  }, []);

  const applyRunSettings = useCallback(() => {
    if (!runDetail) return;
    if (runDetail.config_id) {
      setPendingOverrides(runDetail.config_overrides || {});
      setConfigId(runDetail.config_id);
    }
    setStartDate(runDetail.start_date ? fmtDay(runDetail.start_date) : "");
    setEndDate(runDetail.end_date ? fmtDay(runDetail.end_date) : "");
  }, [runDetail]);

  useEffect(() => {
    if (pendingOverrides && configBase && runDetail && runDetail.config_id === configId) {
      const next = { ...configDraft };
      Object.entries(pendingOverrides).forEach(([key, value]) => {
        if (configTypes[key] === "number") {
          next[key] = value === null || value === undefined ? "" : String(value);
        } else if (configTypes[key] === "boolean") {
          next[key] = Boolean(value);
        } else {
          next[key] = value;
        }
      });
      setConfigDraft(next);
      setPendingOverrides(null);
    }
  }, [pendingOverrides, configBase, configId, configDraft, configTypes, runDetail]);

  const runBacktest = useCallback(async () => {
    if (!configId) {
      setError("Pick a config before running a backtest.");
      return;
    }
    if (startDate && endDate && new Date(startDate) > new Date(endDate)) {
      setError("Start date must be on or before end date.");
      return;
    }
    const { payload, invalid } = buildConfigPayload();
    if (invalid.length) {
      setError(`Fix config values for: ${invalid.join(", ")}`);
      return;
    }
    setStarting(true);
    setError("");
    try {
      const body = {
        config_id: configId,
        config_overrides: payload,
      };
      if (startDate) {
        body.start_date = startDate;
      }
      if (endDate) {
        body.end_date = endDate;
      }
      if (slippageOverride !== "") {
        const num = Number(slippageOverride);
        if (Number.isNaN(num)) {
          throw new Error("Slippage override must be numeric.");
        }
        body.slippage_override = num;
      }
      const res = await fetch(`${API_BASE}/api/backtests/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      await fetchStatus();
      await fetchHistory();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setStarting(false);
    }
  }, [buildConfigPayload, configId, endDate, fetchHistory, fetchStatus, slippageOverride, startDate]);

  useEffect(() => {
    fetchConfigs();
    fetchHistory();
    fetchStatus();
  }, [fetchConfigs, fetchHistory, fetchStatus]);

  useEffect(() => {
    if (configId) {
      fetchConfigDetail(configId);
    }
  }, [configId, fetchConfigDetail]);

  useEffect(() => {
    if (selectedRunId) {
      fetchRunDetail(selectedRunId);
    }
  }, [selectedRunId, fetchRunDetail]);

  useEffect(() => {
    setTradeSort({});
    setTradesPage(1);
  }, [selectedRunId]);

  useEffect(() => {
    if (!runDetail) return;
    const previews = runDetail.trades_preview || {};
    let maxPages = 1;
    Object.values(previews).forEach((info) => {
      const rows = info?.rows || 0;
      const pages = tradesLimit ? Math.max(1, Math.ceil(rows / tradesLimit)) : 1;
      if (pages > maxPages) {
        maxPages = pages;
      }
    });
    if (tradesPage > maxPages) {
      setTradesPage(maxPages);
    }
  }, [runDetail, tradesLimit, tradesPage]);

  useEffect(() => {
    if (status?.running) {
      const id = setInterval(() => {
        fetchStatus();
        fetchHistory();
      }, 5000);
      return () => clearInterval(id);
    }
    return undefined;
  }, [status?.running, fetchStatus, fetchHistory]);

  const configKeys = useMemo(() => Object.keys(configDraft || {}), [configDraft]);
  const numericConfigKeys = useMemo(
    () =>
      configKeys
        .filter((key) => key !== "name" && configTypes[key] === "number")
        .sort((a, b) => a.localeCompare(b)),
    [configKeys, configTypes]
  );
  const booleanConfigKeys = useMemo(
    () =>
      configKeys
        .filter((key) => key !== "name" && configTypes[key] === "boolean")
        .sort((a, b) => a.localeCompare(b)),
    [configKeys, configTypes]
  );

  const activeRunSummary = runDetail?.summary || [];
  const tradesPreview = runDetail?.trades_preview || {};
  const pnlCurves = runDetail?.pnl_curves || {};

  const pnlChartData = useMemo(() => {
    const curves = runDetail?.pnl_curves || {};
    const out = {};
    Object.entries(curves).forEach(([cfg, curve]) => {
      if (!curve) return;
      const dailyMap = {};
      (curve.daily || []).forEach((row) => {
        if (row?.date) {
          dailyMap[row.date] = Number(row.pnl);
        }
      });
      const cumulativeMap = {};
      (curve.cumulative || []).forEach((row) => {
        if (row?.date) {
          cumulativeMap[row.date] = Number(row.pnl);
        }
      });
      const dates = Array.from(new Set([...Object.keys(dailyMap), ...Object.keys(cumulativeMap)])).filter(Boolean);
      dates.sort((a, b) => new Date(a) - new Date(b));
      out[cfg] = dates.map((date) => ({
        date,
        daily: Number.isFinite(dailyMap[date]) ? dailyMap[date] : null,
        cumulative: Number.isFinite(cumulativeMap[date]) ? cumulativeMap[date] : null,
      }));
    });
    return out;
  }, [runDetail?.pnl_curves]);

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 2.5 }}>
      <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ xs: "flex-start", md: "center" }} spacing={2}>
        <Box>
          <Typography variant="h5" gutterBottom>
            Backtests
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Run the daily momentum backtest with any config, tweak parameters, and recall prior runs plus their settings.
          </Typography>
        </Box>
        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="Refresh status">
            <span>
              <IconButton onClick={fetchStatus} disabled={loadingStatus}>
                <Refresh fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Reload history">
            <span>
              <IconButton onClick={fetchHistory} disabled={loadingHistory}>
                <History fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        </Stack>
      </Stack>

      <Card variant="outlined">
        <CardContent sx={{ display: "grid", gap: 2 }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ xs: "stretch", md: "center" }}>
            <TextField
              select
              label="Config preset"
              value={configId}
              onChange={(e) => setConfigId(e.target.value)}
              sx={{ minWidth: 220 }}
              helperText={configBase?.name ? `Loaded ${configBase.name}` : "Pick a config to edit"}
            >
              {configs.length === 0 ? (
                <MenuItem disabled value="">
                  No configs found
                </MenuItem>
              ) : (
                configs.map((config) => (
                  <MenuItem key={config.id} value={config.id}>
                    {config.label || config.id}
                  </MenuItem>
                ))
              )}
            </TextField>
            <TextField
              label="Slippage override ($/side)"
              value={slippageOverride}
              onChange={(e) => setSlippageOverride(e.target.value)}
              placeholder="0.03"
              type="number"
              sx={{ maxWidth: 200 }}
              inputProps={{ step: "any" }}
              helperText="Optional override passed to the tester"
            />
            <TextField
              label="Start date"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ maxWidth: 200 }}
              helperText="Optional backtest start"
            />
            <TextField
              label="End date"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              InputLabelProps={{ shrink: true }}
              sx={{ maxWidth: 200 }}
              helperText="Optional backtest end"
            />
            <Stack direction="row" spacing={1}>
              <Button
                variant="contained"
                startIcon={<PlayCircle />}
                onClick={runBacktest}
                disabled={starting}
              >
                {starting ? "Starting…" : "Run Backtest"}
              </Button>
            </Stack>
            <Box sx={{ flexGrow: 1 }} />
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip label={status?.running ? "Running" : "Idle"} color={status?.running ? "success" : "default"} />
              {status?.id && <Chip label={`Run ${status.id}`} />}
              {status?.config_id && <Chip label={`Config ${status.config_id}`} />}
              {loadingStatus && <CircularProgress size={20} />}
            </Stack>
          </Stack>
          {status?.error && <Alert severity="error">{status.error}</Alert>}
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent sx={{ display: "grid", gap: 2 }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ xs: "stretch", md: "center" }}>
            <Typography variant="subtitle1">Config overrides</Typography>
            <Button variant="outlined" onClick={resetConfigDraft} disabled={!configBase}>
              Reset values
            </Button>
            <Box sx={{ flexGrow: 1 }} />
            {(loadingStatus || loadingRunDetail) && <CircularProgress size={20} />}
          </Stack>
          <Typography variant="body2" color="text.secondary">
            Adjust any config field before launching the backtest. Values are sent as overrides for <code>{configId || "selected config"}</code>.
          </Typography>
          {numericConfigKeys.length === 0 ? (
            <Alert severity="info">No numeric config values loaded.</Alert>
          ) : (
            <Box
              sx={{
                display: "grid",
                gap: 2,
                gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
              }}
            >
              {numericConfigKeys.map((key) => (
                <TextField
                  key={key}
                  label={renderConfigLabel(key)}
                  type="number"
                  value={configDraft[key] ?? ""}
                  onChange={(e) => handleConfigValueChange(key, e.target.value)}
                  inputProps={{ step: "any" }}
                  fullWidth
                />
              ))}
            </Box>
          )}
          {booleanConfigKeys.length > 0 && (
            <>
              <Divider />
              <Box
                sx={{
                  display: "grid",
                  gap: 1,
                  gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
                }}
              >
                {booleanConfigKeys.map((key) => (
                  <FormControlLabel
                    key={key}
                    control={
                      <Switch
                        checked={Boolean(configDraft[key])}
                        onChange={(e) => handleConfigToggle(key, e.target.checked)}
                      />
                    }
                    label={renderConfigLabel(key)}
                  />
                ))}
              </Box>
            </>
          )}
        </CardContent>
      </Card>

      {error && <Alert severity="error">{error}</Alert>}

      <Card variant="outlined">
        <CardContent sx={{ display: "grid", gap: 1.5 }}>
          <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" spacing={1} alignItems={{ xs: "flex-start", md: "center" }}>
            <Typography variant="subtitle1">Backtest history</Typography>
            {loadingHistory && <CircularProgress size={20} />}
          </Stack>
          {history.length === 0 ? (
            <Alert severity="info">No prior runs yet.</Alert>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Run</TableCell>
                  <TableCell>Config</TableCell>
                  <TableCell>Range</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Started</TableCell>
                  <TableCell align="right">Duration</TableCell>
                  <TableCell align="right">Total P/L</TableCell>
                  <TableCell align="right">Sharpe</TableCell>
                  <TableCell align="right">Max DD</TableCell>
                  <TableCell align="right">Trades</TableCell>
                  <TableCell />
                </TableRow>
              </TableHead>
              <TableBody>
                {history.map((run) => {
                  const summaryRow = (run.summary && run.summary[0]) || {};
                  const rangeLabel =
                    run.start_date || run.end_date
                      ? `${fmtDay(run.start_date) !== "—" ? fmtDay(run.start_date) : "—"} → ${
                          fmtDay(run.end_date) !== "—" ? fmtDay(run.end_date) : "—"
                        }`
                      : "—";
                  return (
                    <TableRow key={run.id} selected={run.id === selectedRunId}>
                      <TableCell>{run.id}</TableCell>
                      <TableCell>{run.config_id || "—"}</TableCell>
                      <TableCell>{rangeLabel}</TableCell>
                      <TableCell>
                        <Chip label={run.status || "unknown"} color={run.status === "succeeded" ? "success" : run.status === "running" ? "info" : run.status === "failed" ? "error" : "default"} size="small" />
                      </TableCell>
                      <TableCell>{fmtDate(run.started_at)}</TableCell>
                      <TableCell align="right">{fmtDuration(run.duration_seconds)}</TableCell>
                      <TableCell align="right">{fmtMoney(summaryRow.total_pnl)}</TableCell>
                      <TableCell align="right">{summaryRow.sharpe != null ? Number(summaryRow.sharpe).toFixed(2) : "—"}</TableCell>
                      <TableCell align="right">{fmtPct(summaryRow.max_drawdown_pct)}</TableCell>
                      <TableCell align="right">{fmtNumber(summaryRow.num_trades)}</TableCell>
                      <TableCell align="right">
                        <Button size="small" onClick={() => setSelectedRunId(run.id)}>
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent sx={{ display: "grid", gap: 1.5 }}>
          <Stack direction={{ xs: "column", md: "row" }} alignItems={{ xs: "flex-start", md: "center" }} spacing={1}>
            <Typography variant="subtitle1">Run details</Typography>
            {runDetail?.id && <Chip label={runDetail.id} />}
            {runDetail?.config_id && <Chip label={`Config ${runDetail.config_id}`} />}
            {loadingRunDetail && <CircularProgress size={18} />}
            <Box sx={{ flexGrow: 1 }} />
            <Button variant="outlined" size="small" disabled={!runDetail} onClick={applyRunSettings}>
              Load settings
            </Button>
          </Stack>
          {runDetail?.status === "failed" && <Alert severity="error">{runDetail.error || "Run failed"}</Alert>}
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            {runDetail?.start_date && <Chip size="small" label={`Start ${fmtDay(runDetail.start_date)}`} />}
            {runDetail?.end_date && <Chip size="small" label={`End ${fmtDay(runDetail.end_date)}`} />}
            {runDetail?.started_at && <Chip size="small" label={`Started ${fmtDate(runDetail.started_at)}`} />}
            {runDetail?.finished_at && <Chip size="small" label={`Finished ${fmtDate(runDetail.finished_at)}`} />}
            {runDetail?.duration_seconds != null && <Chip size="small" label={`Duration ${fmtDuration(runDetail.duration_seconds)}`} />}
          </Stack>
          <SummaryTable records={activeRunSummary} />
          {pnlCurves && Object.keys(pnlCurves).length > 0 && (
            <>
              <Divider />
              <Typography variant="subtitle2">P&amp;L timeline</Typography>
              {Object.entries(pnlCurves).map(([cfg, curve]) => {
                const data = pnlChartData[cfg] || [];
                return (
                  <Box key={cfg} sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                      {cfg} — {data.length} day{data.length === 1 ? "" : "s"}
                    </Typography>
                  {data.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                      No P&amp;L data available.
                    </Typography>
                  ) : (
                    <Box sx={{ width: "100%", maxWidth: 1040, mx: "auto" }}>
                      <Box sx={{ width: "100%", height: 260, pr: 1 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="date"
                              minTickGap={20}
                              tickFormatter={(val) =>
                                new Date(val).toLocaleDateString(undefined, { month: "short", day: "numeric" })
                              }
                            />
                            <YAxis tickFormatter={(v) => `$${Number(v).toFixed(0)}`} width={70} />
                            <RechartsTooltip
                              labelFormatter={(val) => new Date(val).toLocaleDateString()}
                              formatter={(value, name, props) => {
                                const dataKey = props?.dataKey || name;
                                const label = dataKey === "daily" ? "Daily P&L" : "Cumulative P&L";
                                return [`$${Number(value ?? 0).toFixed(2)}`, label];
                              }}
                            />
                            <Legend />
                            <Bar dataKey="daily" name="Daily P&L" barSize={18}>
                              {data.map((entry, idx) => (
                                <Cell key={entry.date || idx} fill={entry.daily >= 0 ? "#2e7d32" : "#c62828"} />
                              ))}
                            </Bar>
                            <Line
                              type="monotone"
                              dataKey="cumulative"
                              name="Cumulative P&L"
                              stroke="#1976d2"
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </Box>
                    </Box>
                  )}
                </Box>
              );
            })}
            </>
          )}

          {runDetail?.trades_preview && Object.keys(runDetail.trades_preview).length > 0 && (
            <>
              <Divider />
            <Stack direction={{ xs: "column", md: "row" }} spacing={1} alignItems={{ xs: "flex-start", md: "center" }} justifyContent="space-between">
              <Typography variant="subtitle2">Trades preview</Typography>
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="caption" color="text.secondary">
                  Rows:
                </Typography>
                  <ButtonGroup size="small" variant="outlined">
                    {[25, 50, 100].map((limit) => (
                      <Button
                        key={limit}
                        variant={tradesLimit === limit ? "contained" : "outlined"}
                        onClick={() => handleTradesLimitChange(limit)}
                        disabled={loadingRunDetail}
                      >
                        {limit}
                      </Button>
                    ))}
                  </ButtonGroup>
                </Stack>
              </Stack>
              {Object.entries(tradesPreview).map(([cfg, info]) => {
                const sortState = tradeSort[cfg];
                const rows = sortPreviewRows(info?.preview || [], sortState);
                const columns = rows?.[0] ? Object.keys(rows[0]) : [];
                const totalRows = info?.rows || 0;
                const totalPages = tradesLimit ? Math.max(1, Math.ceil(totalRows / tradesLimit)) : 1;
                const currentPage = Math.min(tradesPage, totalPages);
                const offset = info?.offset || Math.max(0, (currentPage - 1) * tradesLimit);
                const startRow = totalRows === 0 ? 0 : offset + 1;
                const endRow = totalRows === 0 ? 0 : Math.min(totalRows, offset + rows.length);
                return (
                  <Box key={cfg} sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                      {cfg} — showing {rows.length} of {totalRows} trades (rows {startRow}–{endRow})
                    </Typography>
                    <TableContainer sx={{ width: "100%", maxWidth: 1040, mx: "auto", overflowX: "auto" }}>
                      <Table size="small" sx={{ minWidth: { xs: 0, md: 720 } }}>
                        <TableHead>
                          <TableRow>
                            {columns.map((col) => {
                              const isActive = sortState?.column === col;
                              return (
                                <TableCell
                                  key={col}
                                  onClick={() => handleTradeSort(cfg, col)}
                                  sx={{ cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" }}
                                >
                                  {formatConfigLabel(col)}
                                  {isActive ? (sortState.direction === "asc" ? " (asc)" : " (desc)") : ""}
                                </TableCell>
                              );
                            })}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {rows && rows.length > 0 ? (
                            rows.map((row, idx) => (
                              <TableRow key={idx}>
                                {columns.map((col) => (
                                  <TableCell key={col}>{row[col]}</TableCell>
                                ))}
                              </TableRow>
                            ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={Math.max(columns.length, 1)}>No trades recorded.</TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    {totalPages > 1 && (
                      <Stack direction="row" spacing={1} alignItems="center" justifyContent="flex-end" sx={{ mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          Page {currentPage} of {totalPages}
                        </Typography>
                        <ButtonGroup size="small" variant="outlined">
                          <Button onClick={() => handleTradesPageChange(1)} disabled={currentPage <= 1}>
                            First
                          </Button>
                          <Button onClick={() => handleTradesPageChange(currentPage - 1)} disabled={currentPage <= 1}>
                            Prev
                          </Button>
                          <Button onClick={() => handleTradesPageChange(currentPage + 1)} disabled={currentPage >= totalPages}>
                            Next
                          </Button>
                          <Button onClick={() => handleTradesPageChange(totalPages)} disabled={currentPage >= totalPages}>
                            Last
                          </Button>
                        </ButtonGroup>
                      </Stack>
                    )}
                  </Box>
                );
              })}
            </>
          )}
        </CardContent>
      </Card>
    </Container>
  );
}
