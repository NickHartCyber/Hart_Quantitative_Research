import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  Container,
  Divider,
  InputAdornment,
  MenuItem,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { Search } from "@mui/icons-material";
import { Link as RouterLink } from "react-router-dom";

const API_BASE = process.env.REACT_APP_API_BASE || "";

const DAY_OPTIONS = [7, 14, 30, 60, 90, 180];
const TYPE_OPTIONS = [
  { value: "all", label: "All types" },
  { value: "purchase", label: "Purchases" },
  { value: "sale", label: "Sales" },
  { value: "exchange", label: "Exchanges" },
  { value: "gift", label: "Gifts" },
  { value: "other", label: "Other" },
];

const PARTY_COLORS = {
  D: "info",
  R: "error",
  I: "warning",
};

const formatDate = (value) => {
  if (!value) return "--";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "--";
  return dt.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
};

const formatTimestamp = (value) => {
  if (!value) return "Not fetched yet";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "Not fetched yet";
  return dt.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
};

const formatAmountRange = (min, max) => {
  const minNum = Number(min);
  const maxNum = Number(max);
  const fmt = (val) =>
    new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(val);
  if (Number.isFinite(minNum) && Number.isFinite(maxNum)) {
    return minNum === maxNum ? fmt(minNum) : `${fmt(minNum)} - ${fmt(maxNum)}`;
  }
  if (Number.isFinite(minNum)) return `${fmt(minNum)}+`;
  if (Number.isFinite(maxNum)) return `Up to ${fmt(maxNum)}`;
  return "--";
};

const normalizeType = (value = "") => {
  const raw = String(value || "").toLowerCase();
  if (!raw) return "Other";
  if (raw.includes("purchase") || raw.includes("buy")) return "Purchase";
  if (raw.includes("sale") || raw.includes("sell")) return "Sale";
  if (raw.includes("exchange")) return "Exchange";
  if (raw.includes("gift")) return "Gift";
  return raw.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
};

const typeMatches = (filter, typeValue) => {
  if (filter === "all") return true;
  const normalized = String(typeValue || "").toLowerCase();
  if (filter === "purchase") return normalized.includes("purchase") || normalized.includes("buy");
  if (filter === "sale") return normalized.includes("sale") || normalized.includes("sell");
  if (filter === "exchange") return normalized.includes("exchange");
  if (filter === "gift") return normalized.includes("gift");
  return !normalized || (!normalized.includes("purchase") && !normalized.includes("sale") && !normalized.includes("exchange") && !normalized.includes("gift"));
};

const typeChipColor = (value) => {
  const normalized = String(value || "").toLowerCase();
  if (normalized.includes("purchase") || normalized.includes("buy")) return "success";
  if (normalized.includes("sale") || normalized.includes("sell")) return "error";
  return "default";
};

const StatCard = ({ label, value, helper }) => (
  <Paper sx={{ p: 2.5, flex: 1, minWidth: 180 }}>
    <Typography variant="overline" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="h5" sx={{ fontWeight: 700 }}>
      {value}
    </Typography>
    {helper ? (
      <Typography variant="body2" color="text.secondary">
        {helper}
      </Typography>
    ) : null}
  </Paper>
);

export default function PoliticianTradesPage() {
  const [trades, setTrades] = useState([]);
  const [days, setDays] = useState(90);
  const [typeFilter, setTypeFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("date");
  const [sortDir, setSortDir] = useState("desc");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [meta, setMeta] = useState({
    fetchedAt: null,
    cached: false,
    stale: false,
    source: "",
    sources: [],
    days: null,
    requestedDays: null,
  });

  const fetchTrades = useCallback(
    async () => {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams({
          days: String(days),
          limit: "250",
        });
        const res = await fetch(`${API_BASE}/api/politician-trades?${params.toString()}`);
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(txt || `HTTP ${res.status}`);
        }
        const data = await res.json();
        const list = Array.isArray(data?.trades) ? data.trades : [];
        setTrades(list);
        setMeta({
          fetchedAt: data?.fetched_at || null,
          cached: Boolean(data?.cached),
          stale: Boolean(data?.stale),
          source: data?.source || "",
          sources: Array.isArray(data?.sources) ? data.sources : [],
          liveError: data?.live_error || "",
          days: Number.isFinite(Number(data?.days)) ? Number(data?.days) : null,
          requestedDays: Number.isFinite(Number(data?.requested_days)) ? Number(data?.requested_days) : null,
        });
      } catch (err) {
        setError(err.message || "Failed to load trades.");
        setTrades([]);
      } finally {
        setLoading(false);
      }
    },
    [days],
  );

  useEffect(() => {
    fetchTrades();
  }, [fetchTrades]);

  const filteredTrades = useMemo(() => {
    const query = search.trim().toLowerCase();
    return trades.filter((trade) => {
      if (!typeMatches(typeFilter, trade?.transaction_type)) return false;
      if (!query) return true;
      const haystack = [
        trade?.politician,
        trade?.ticker,
        trade?.asset_name,
        trade?.comment,
        trade?.owner,
        trade?.state,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [trades, search, typeFilter]);

  const sortedTrades = useMemo(() => {
    const dir = sortDir === "asc" ? 1 : -1;
    const getDate = (trade) => {
      const value = trade?.transaction_date || trade?.report_date || trade?.filed_date;
      const ts = value ? new Date(value).getTime() : 0;
      return Number.isNaN(ts) ? 0 : ts;
    };
    const getAmount = (trade) => {
      const max = Number(trade?.amount_max);
      const min = Number(trade?.amount_min);
      if (Number.isFinite(max)) return max;
      if (Number.isFinite(min)) return min;
      return 0;
    };
    const getValue = (trade) => {
      if (sortBy === "politician") return String(trade?.politician || "").toLowerCase();
      if (sortBy === "ticker") return String(trade?.ticker || "").toLowerCase();
      if (sortBy === "type") return String(trade?.transaction_type || "").toLowerCase();
      if (sortBy === "amount") return getAmount(trade);
      return getDate(trade);
    };
    return [...filteredTrades].sort((a, b) => {
      const va = getValue(a);
      const vb = getValue(b);
      if (va === vb) return 0;
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
      return va > vb ? dir : -dir;
    });
  }, [filteredTrades, sortBy, sortDir]);

  const summary = useMemo(() => {
    const tickers = new Set();
    const politicians = new Set();
    let buys = 0;
    let sells = 0;
    trades.forEach((trade) => {
      if (trade?.ticker) tickers.add(String(trade.ticker).toUpperCase());
      if (trade?.politician) politicians.add(trade.politician);
      const normalized = String(trade?.transaction_type || "").toLowerCase();
      if (normalized.includes("purchase") || normalized.includes("buy")) buys += 1;
      if (normalized.includes("sale") || normalized.includes("sell")) sells += 1;
    });
    return {
      total: trades.length,
      tickers: tickers.size,
      politicians: politicians.size,
      buys,
      sells,
    };
  }, [trades]);

  const lookbackLabel = meta?.days || days;

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortDir("asc");
    }
  };

  const sourceLabel = meta?.sources?.length
    ? meta.sources.join(", ")
    : meta?.source
      ? meta.source
      : "unknown";
  const sourceText = `Data sources: ${sourceLabel}.`;

  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        <Paper
          sx={{
            p: 3,
            background: "linear-gradient(135deg, rgba(15,39,71,0.95), rgba(31,59,99,0.95))",
            color: "#f6f9fc",
          }}
        >
          <Stack spacing={1}>
            <Typography variant="h4" sx={{ fontWeight: 700 }}>
              Politician Trades
            </Typography>
            <Typography variant="body1" sx={{ maxWidth: 680 }}>
              Track recent congressional stock activity. Filter by ticker, member, or transaction type to spot the
              freshest disclosures.
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.85 }}>
              {meta?.stale ? `Cached snapshot (live refresh failed). ${sourceText}` : sourceText} Last updated{" "}
              {formatTimestamp(meta?.fetchedAt)}.
            </Typography>
          </Stack>
        </Paper>

        {error ? (
          <Alert severity="error">{error}</Alert>
        ) : meta?.stale ? (
          <Alert severity="warning">{meta?.liveError || "Showing cached data due to a live fetch issue."}</Alert>
        ) : null}

        <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
          <StatCard label="Trades pulled" value={summary.total} helper={`Past ${lookbackLabel} days`} />
          <StatCard label="Unique tickers" value={summary.tickers} />
          <StatCard label="Unique politicians" value={summary.politicians} />
          <StatCard label="Buys vs sells" value={`${summary.buys} / ${summary.sells}`} />
        </Stack>

        <Paper sx={{ p: 2.5 }}>
          <Stack spacing={2}>
            <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
              <TextField
                select
                label="Lookback"
                size="small"
                value={days}
                onChange={(event) => setDays(Number(event.target.value))}
                sx={{ minWidth: 160 }}
              >
                {DAY_OPTIONS.map((value) => (
                  <MenuItem key={value} value={value}>
                    {value} days
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                select
                label="Type"
                size="small"
                value={typeFilter}
                onChange={(event) => setTypeFilter(event.target.value)}
                sx={{ minWidth: 160 }}
              >
                {TYPE_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                label="Search ticker, member, asset"
                size="small"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search color="action" />
                    </InputAdornment>
                  ),
                }}
                sx={{ flex: 1, minWidth: 240 }}
              />
            </Stack>
            <Divider />
            <Typography variant="body2" color="text.secondary">
              Showing {sortedTrades.length} of {trades.length} trades.
            </Typography>
          </Stack>
        </Paper>

        <Paper>
          <TableContainer sx={{ maxHeight: 620 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell sortDirection={sortBy === "date" ? sortDir : false}>
                    <TableSortLabel
                      active={sortBy === "date"}
                      direction={sortBy === "date" ? sortDir : "asc"}
                      onClick={() => handleSort("date")}
                    >
                      Date
                    </TableSortLabel>
                  </TableCell>
                  <TableCell sortDirection={sortBy === "politician" ? sortDir : false}>
                    <TableSortLabel
                      active={sortBy === "politician"}
                      direction={sortBy === "politician" ? sortDir : "asc"}
                      onClick={() => handleSort("politician")}
                    >
                      Politician
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>Party</TableCell>
                  <TableCell sortDirection={sortBy === "ticker" ? sortDir : false}>
                    <TableSortLabel
                      active={sortBy === "ticker"}
                      direction={sortBy === "ticker" ? sortDir : "asc"}
                      onClick={() => handleSort("ticker")}
                    >
                      Ticker
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>Asset</TableCell>
                  <TableCell sortDirection={sortBy === "type" ? sortDir : false}>
                    <TableSortLabel
                      active={sortBy === "type"}
                      direction={sortBy === "type" ? sortDir : "asc"}
                      onClick={() => handleSort("type")}
                    >
                      Type
                    </TableSortLabel>
                  </TableCell>
                  <TableCell sortDirection={sortBy === "amount" ? sortDir : false}>
                    <TableSortLabel
                      active={sortBy === "amount"}
                      direction={sortBy === "amount" ? sortDir : "asc"}
                      onClick={() => handleSort("amount")}
                    >
                      Amount
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>Owner</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={8}>
                      <Stack alignItems="center" spacing={1} sx={{ py: 4 }}>
                        <CircularProgress size={28} />
                        <Typography variant="body2" color="text.secondary">
                          Loading trades...
                        </Typography>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ) : sortedTrades.length ? (
                  sortedTrades.map((trade, idx) => {
                    const dateLabel = formatDate(trade?.transaction_date || trade?.report_date);
                    const party = String(trade?.party || "").toUpperCase();
                    const partyColor = PARTY_COLORS[party] || "default";
                    const ticker = String(trade?.ticker || "").trim();
                    return (
                      <TableRow key={`${trade?.source_id || idx}`} hover>
                        <TableCell>{dateLabel}</TableCell>
                        <TableCell>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {trade?.politician || "--"}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {(trade?.state || "").toUpperCase()} {trade?.chamber ? `| ${trade.chamber}` : ""}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {party ? (
                            <Chip label={party} size="small" color={partyColor} />
                          ) : (
                            <Chip label="--" size="small" variant="outlined" />
                          )}
                        </TableCell>
                        <TableCell>
                          {ticker ? (
                            <Chip
                              label={ticker.toUpperCase()}
                              size="small"
                              variant="outlined"
                              clickable
                              component={RouterLink}
                              to={`/ticker?ticker=${encodeURIComponent(ticker)}`}
                            />
                          ) : (
                            <Chip label="--" size="small" variant="outlined" />
                          )}
                        </TableCell>
                        <TableCell>
                          {trade?.asset_name ? (
                            <Tooltip title={trade.asset_name} placement="top">
                              <Typography variant="body2" sx={{ maxWidth: 220 }} noWrap>
                                {trade.asset_name}
                              </Typography>
                            </Tooltip>
                          ) : (
                            "--"
                          )}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={normalizeType(trade?.transaction_type)}
                            size="small"
                            color={typeChipColor(trade?.transaction_type)}
                          />
                        </TableCell>
                        <TableCell>{formatAmountRange(trade?.amount_min, trade?.amount_max)}</TableCell>
                        <TableCell>
                          <Typography variant="body2">{trade?.owner || "--"}</Typography>
                        </TableCell>
                      </TableRow>
                    );
                  })
                ) : (
                  <TableRow>
                    <TableCell colSpan={8}>
                      <Box sx={{ py: 4, textAlign: "center" }}>
                        <Typography variant="body1">No trades match this filter.</Typography>
                        <Typography variant="body2" color="text.secondary">
                          Try widening the lookback window or clearing the search.
                        </Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Stack>
    </Container>
  );
}
