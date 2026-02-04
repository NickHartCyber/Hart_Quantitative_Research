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
  IconButton,
  FormControlLabel,
  MenuItem,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { Bolt, InfoOutlined, PlayCircle, Refresh, Stop } from "@mui/icons-material";

const API_BASE = process.env.REACT_APP_API_BASE || "";

const fmtMoney = (val) => {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  const num = Number(val);
  const sign = num >= 0 ? "" : "-";
  return `${sign}$${Math.abs(num).toFixed(2)}`;
};

const fmtNumber = (val) => {
  if (val === null || val === undefined || Number.isNaN(val)) return "—";
  return Number(val).toLocaleString();
};

const formatConfigLabel = (key) =>
  String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());

const CONFIG_DESCRIPTIONS = {
  ema_fast: "Fast EMA length for short-term trend. Example: 9 vs 20/50; BUY requires ema_fast > ema_med > ema_slow. Smaller = more reactive entries, bigger = slower/safer.",
  ema_med: "Medium EMA in the trend stack. Example: 20 vs 9/50; tightening this (e.g., 15) demands tighter alignment before buying.",
  ema_slow: "Slow EMA anchor for overall trend. Example: 50; larger pushes buys later into stronger trends.",
  min_adx: "Minimum ADX(14) to accept a breakout. Example: 14 means light trend is OK; 20+ blocks weak trends and reduces buys.",
  thrust_atr_mult: "Signal bar range vs ATR gate. Example: 1.3 means bar range must be ≥1.3× ATR; higher = need stronger thrust to buy.",
  min_body_frac: "Body fraction of bar range. Example: 0.5 keeps bars with at least half body (fewer wick-only spikes); higher blocks more setups.",
  lookback_break: "Bars used to define breakout high. Example: 3 means break prior 3-bar high; larger waits for bigger bases, fewer entries.",
  entry_buffer_atr: "ATR buffer above breakout trigger. Example: 0.03 means stop trigger = level + 0.03×ATR; larger delays fills and reduces false breaks.",
  initial_stop_atr: "Initial stop distance in ATR. Example: 0.8 places stop 0.8×ATR below entry; wider stops reduce stop-outs but lower size/R.",
  trail_atr_mult: "ATR multiple for trailing stop after move in your favor. Example: 1.1 trails at high − 1.1×ATR; larger = looser trail.",
  take_profit_R: "Take-profit at R multiples (R = entry−initial stop). Example: 2.0 exits near 2R; higher chases bigger winners, fewer hits.",
  max_hold_minutes: "Time stop; closes any position after this many minutes. Example: 90 closes late lingerers; shorter = faster recycle.",
  rvol_recent: "Recent volume window for rVol. Example: 3 bars average vs baseline; higher smooths spikes.",
  rvol_window_baseline: "Baseline volume window (excludes recent window). Example: 20; larger demands sustained participation to buy.",
  min_dollar_bar_reg: "Min dollar/bar (price×volume) during RTH. Example: 25k; raises reduce thin names; buy blocked if below.",
  min_dollar_bar_pre: "Min dollar/bar during premarket. Example: 10k; prevents illiquid premarket fills.",
  allow_premarket: "Allow signals and trading in premarket. OFF blocks buys before 9:30; ON can enter early but with thinner books.",
  enable_cost_guards: "Master toggle for cost guards (price floor, ATR vs slippage, expected move vs cost, min TP). OFF disables those checks.",
  min_price_dollars: "Skip names below this price. Example: 5.0 removes sub-$5 tickers; fewer cheap spikes, fewer total entries.",
  atr_vs_slip_min_mult: "Require ATR ≥ this × per-side slippage. Example: 5.0 means ATR must be ≥5× assumed slippage; blocks trades with poor reward vs slip.",
  exp_move_vs_cost_min_mult: "Require expected move ≥ this × round-trip slippage. Example: 4.0; blocks trades where cost overwhelms target.",
  take_profit_min_cents: "Floor for take-profit in cents. Example: 10 sets TP ≥ $0.10 above entry even if R target is smaller; 0 disables.",
  use_stop_limit: "Use stop-limit entry instead of stop-market. ON caps entry slippage but risks no fill on fast moves.",
  stop_limit_offset_cents: "Limit cap above trigger for stop-limit. Example: 2 sets limit = stop + $0.02; larger fills more often but worse price.",
  slippage_assumed: "Assumed per-side slippage ($) for guards and logging. Example: 0.03; raising makes cost gates stricter (fewer buys).",
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

const Pill = ({ label, color = "default" }) => (
  <Chip label={label} size="small" color={color} variant="filled" />
);

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

export default function PartyTimePage() {
  const [algos, setAlgos] = useState([]);
  const [selected, setSelected] = useState("");
  const [rows, setRows] = useState([]);
  const [status, setStatus] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [configs, setConfigs] = useState([]);
  const [configId, setConfigId] = useState("");
  const [configBase, setConfigBase] = useState(null);
  const [configDraft, setConfigDraft] = useState({});
  const [configTypes, setConfigTypes] = useState({});
  const [loadingConfigs, setLoadingConfigs] = useState(false);
  const [loadingConfigDetail, setLoadingConfigDetail] = useState(false);
  const [sortBy, setSortBy] = useState("symbol");
  const [sortDir, setSortDir] = useState("asc");

  const activeAlgo = useMemo(() => {
    if (!status?.algo) return null;
    return algos.find((a) => a.id === status.algo);
  }, [status, algos]);

  const fetchAlgos = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/party-time/algos`);
      if (!res.ok) {
        throw new Error(`Failed to load algos (${res.status})`);
      }
      const data = await res.json();
      const list = data.algos || [];
      setAlgos(list);
      if (!selected && list.length > 0) {
        setSelected(list[0].id);
      }
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [selected]);

  const fetchStatus = useCallback(async () => {
    setLoadingStatus(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/party-time/status`);
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setStatus(data);
      setRows(Array.isArray(data?.rows) ? data.rows : []);
    } catch (e) {
      setError(e.message || String(e));
      setRows([]);
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  const fetchConfigs = useCallback(async () => {
    setLoadingConfigs(true);
    try {
      const res = await fetch(`${API_BASE}/api/party-time/configs`);
      if (!res.ok) {
        throw new Error(`Failed to load configs (${res.status})`);
      }
      const data = await res.json();
      const list = data.configs || [];
      setConfigs(list);
      if (!configId && list.length > 0) {
        const preferred = list.find((item) => item.id === "baseline")?.id || list[0].id;
        setConfigId(preferred);
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoadingConfigs(false);
    }
  }, [configId]);

  const fetchConfigDetail = useCallback(async (id) => {
    if (!id) return;
    setLoadingConfigDetail(true);
    try {
      const res = await fetch(`${API_BASE}/api/party-time/configs/${id}`);
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
    } finally {
      setLoadingConfigDetail(false);
    }
  }, []);

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

  const runSelected = useCallback(async () => {
    if (!selected) {
      setError("Select an algorithm before starting it.");
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
      const body = { algo: selected };
      if (configId) {
        body.config_id = configId;
      }
      if (payload && Object.keys(payload).length > 0) {
        body.config_overrides = payload;
      }
      const res = await fetch(`${API_BASE}/api/party-time/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      await fetchStatus();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setStarting(false);
    }
  }, [selected, buildConfigPayload, configId, fetchStatus]);

  const stopAlgo = useCallback(async () => {
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/party-time/stop`, { method: "POST" });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      await fetchStatus();
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [fetchStatus]);

  useEffect(() => {
    fetchAlgos();
    fetchStatus();
    fetchConfigs();
  }, [fetchAlgos, fetchStatus, fetchConfigs]);

  useEffect(() => {
    if (configId) {
      fetchConfigDetail(configId);
    }
  }, [configId, fetchConfigDetail]);

  useEffect(() => {
    if (status?.running) {
      const id = setInterval(fetchStatus, 4000);
      return () => clearInterval(id);
    }
    return undefined;
  }, [status?.running, fetchStatus]);

  const lastUpdated = status?.last_update
    ? new Date(status.last_update * 1000).toLocaleTimeString()
    : null;
  const startedAt = status?.started_at
    ? new Date(status.started_at * 1000).toLocaleTimeString()
    : null;
  const activeConfigLabel = status?.config?.name || status?.config_id || null;

  const columns = [
    { id: "symbol", label: "Symbol", align: "left" },
    { id: "last_price", label: "Price", align: "right", numeric: true },
    { id: "net_change", label: "Change", align: "right", numeric: true },
    { id: "net_change_pct", label: "% Change", align: "right", numeric: true },
    { id: "shares_held", label: "Shares Held", align: "right", numeric: true },
    { id: "open_pnl", label: "Open P/L", align: "right", numeric: true },
    { id: "day_pnl", label: "Day P/L", align: "right", numeric: true },
    { id: "volume", label: "Volume", align: "right", numeric: true },
    { id: "shares_outstanding", label: "Shares Outstanding", align: "right", numeric: true },
    { id: "rating", label: "Rating", align: "right", numeric: true },
  ];

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortDir("asc");
    }
  };

  const sortedRows = useMemo(() => {
    const getVal = (row, key) => {
      const v = row?.[key];
      if (v === null || v === undefined) return null;
      const num = Number(v);
      if (!Number.isNaN(num) && typeof v !== "string") return num;
      return typeof v === "string" ? v.toUpperCase() : num;
    };

    const dir = sortDir === "desc" ? -1 : 1;
    return [...rows].sort((a, b) => {
      const va = getVal(a, sortBy);
      const vb = getVal(b, sortBy);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === "string" && typeof vb === "string") {
        return va.localeCompare(vb) * dir;
      }
      const na = Number(va);
      const nb = Number(vb);
      if (Number.isNaN(na) || Number.isNaN(nb)) return 0;
      return (na - nb) * dir;
    });
  }, [rows, sortBy, sortDir]);

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

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 2.5 }}>
      <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ xs: "flex-start", md: "center" }} spacing={2}>
        <Box>
          <Typography variant="h5" gutterBottom>
            Automated Quant Execution
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Choose a <code>party_time_*</code> algorithm, launch it, and watch the live queue of tickers it is working through.
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
        </Stack>
      </Stack>

      <Card variant="outlined">
        <CardContent sx={{ display: "grid", gap: 2 }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ xs: "stretch", md: "center" }}>
            <TextField
              select
              label="Algorithm"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
              sx={{ minWidth: 260 }}
              helperText={activeAlgo?.description || "Pick which party_time orchestration to run"}
            >
              {algos.map((algo) => (
                <MenuItem key={algo.id} value={algo.id}>
                  {algo.label || algo.id}
                </MenuItem>
              ))}
            </TextField>
            <Stack direction="row" spacing={1}>
              <Button
                variant="contained"
                startIcon={<PlayCircle />}
                onClick={runSelected}
                disabled={starting}
              >
                {starting ? "Starting…" : "Execute"}
              </Button>
              <Button
                variant="outlined"
                color="error"
                startIcon={<Stop />}
                onClick={stopAlgo}
                disabled={loadingStatus}
              >
                Stop
              </Button>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Pill
                label={status?.running ? "Running" : "Idle"}
                color={status?.running ? "success" : "default"}
              />
              {status?.algo && <Pill label={activeAlgo?.label || status.algo} color="info" />}
              {status?.last_error && <Pill label="Error" color="error" />}
            </Stack>
            <Box sx={{ flexGrow: 1 }} />
            {status?.count !== undefined && (
              <Typography variant="body2" color="text.secondary">
                Queue size: <strong>{status.count}</strong>
              </Typography>
            )}
          </Stack>
          <Divider />
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <Chip icon={<Bolt />} label={`Status: ${status?.status || "unknown"}`} />
            {startedAt && <Chip label={`Started at ${startedAt}`} />}
            {lastUpdated && <Chip label={`Last update ${lastUpdated}`} />}
            {activeConfigLabel && <Chip label={`Config ${activeConfigLabel}`} />}
          </Stack>
          {status?.last_error && <Alert severity="error">{status.last_error}</Alert>}
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent sx={{ display: "grid", gap: 2 }}>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ xs: "stretch", md: "center" }}>
            <TextField
              select
              label="Config preset"
              value={configId}
              onChange={(e) => setConfigId(e.target.value)}
              sx={{ minWidth: 220 }}
              helperText={configBase?.name ? `Loaded ${configBase.name}` : "Pick a baseline config to edit"}
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
            <Button variant="outlined" onClick={resetConfigDraft} disabled={!configBase}>
              Reset values
            </Button>
            <Box sx={{ flexGrow: 1 }} />
            {(loadingConfigs || loadingConfigDetail) && <CircularProgress size={20} />}
          </Stack>
          <Typography variant="body2" color="text.secondary">
            Adjust config numbers before execution. These settings apply to the Momentum Breakout party-time run.
          </Typography>
          {numericConfigKeys.length === 0 && !loadingConfigDetail ? (
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
        <CardContent>
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="subtitle1">Live queue</Typography>
            {loadingStatus && <CircularProgress size={20} />}
          </Stack>
          {sortedRows.length === 0 ? (
            <Alert severity="info">No tickers in the queue yet.</Alert>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  {columns.map((col) => (
                    <TableCell key={col.id} align={col.align}>
                      <TableSortLabel
                        active={sortBy === col.id}
                        direction={sortBy === col.id ? sortDir : "asc"}
                        onClick={() => handleSort(col.id)}
                      >
                        {col.label}
                      </TableSortLabel>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {sortedRows.map((r) => {
                  const changeColor = r.net_change > 0 ? "success.main" : r.net_change < 0 ? "error.main" : "text.secondary";
                  const pnlColor = (val) =>
                    val > 0 ? "success.main" : val < 0 ? "error.main" : "text.secondary";
                  return (
                    <TableRow key={r.symbol}>
                      <TableCell>
                        <Typography fontWeight={700}>{r.symbol}</Typography>
                      </TableCell>
                      <TableCell align="right">{fmtMoney(r.last_price)}</TableCell>
                      <TableCell align="right" sx={{ color: changeColor }}>
                        {fmtMoney(r.net_change)}
                      </TableCell>
                      <TableCell align="right" sx={{ color: changeColor }}>
                        {r.net_change_pct != null ? `${Number(r.net_change_pct).toFixed(2)}%` : "—"}
                      </TableCell>
                      <TableCell align="right">{fmtNumber(r.shares_held)}</TableCell>
                      <TableCell align="right" sx={{ color: pnlColor(r.open_pnl) }}>
                        {fmtMoney(r.open_pnl)}
                      </TableCell>
                      <TableCell align="right" sx={{ color: pnlColor(r.day_pnl) }}>
                        {fmtMoney(r.day_pnl)}
                      </TableCell>
                      <TableCell align="right">{fmtNumber(r.volume)}</TableCell>
                      <TableCell align="right">{fmtNumber(r.shares_outstanding)}</TableCell>
                      <TableCell align="right">
                        {r.rating != null ? r.rating : "—"}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </Container>
  );
}
