import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link as RouterLink, useLocation } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  FormControl,
  InputLabel,
  Link,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { Article, OpenInNew, Search, Send, ShowChart, SyncAlt } from "@mui/icons-material";
import {
  Bar,
  Cell,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  usePlotArea,
} from "recharts";

const API_BASE = process.env.REACT_APP_API_BASE || "";

const TIMEFRAME_OPTIONS = [
  { tf: "1D", label: "1 Day" },
  { tf: "5D", label: "5 Day" },
  { tf: "1M", label: "1 Month" },
  { tf: "6M", label: "6 Month" },
  { tf: "1Y", label: "1 Year" },
  { tf: "5Y", label: "5 Year" },
  { tf: "10Y", label: "10 Year" },
];

const CHART_TYPE_OPTIONS = [
  { value: "line", label: "Line Graph" },
  { value: "candle", label: "Candle Graph" },
];

const FUNDAMENTAL_FIELDS = [
  { key: "price", label: "Last Price", type: "currency" },
  { key: "market_cap", label: "Market Cap", type: "compactCurrency" },
  { key: "sector", label: "Sector", type: "text" },
  { key: "pe_ratio", label: "P/E (TTM)", type: "number" },
  { key: "forward_pe", label: "Forward P/E", type: "number" },
  { key: "peg_ratio", label: "PEG Ratio", type: "number" },
  { key: "eps_ttm", label: "EPS (TTM)", type: "number" },
  { key: "roe", label: "Return on Equity", type: "percent" },
  { key: "debt_to_equity", label: "Debt to Equity", type: "number" },
  { key: "dividend_yield", label: "Dividend Yield", type: "percent" },
  { key: "profit_margin", label: "Profit Margin", type: "percent" },
  { key: "book_value", label: "Book Value / Share", type: "currency" },
  { key: "fifty_two_week_high", label: "52w High", type: "currency" },
  { key: "fifty_two_week_low", label: "52w Low", type: "currency" },
  { key: "beta", label: "Beta", type: "number" },
  { key: "shares_outstanding", label: "Shares Outstanding", type: "compactNumber" },
];

const SECTOR_METRIC_LABELS = {
  pe_ratio: "P/E (TTM)",
  forward_pe: "Forward P/E",
  profit_margin: "Profit Margin",
  roe: "Return on Equity",
};

function compactCurrency(value, currency = "USD") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatCurrency(value, currency = "USD") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const num = Number(value);
  // Some providers already send percent values (e.g., 12.3) vs. ratios (0.123)
  const ratio = Math.abs(num) <= 1 ? num * 100 : num;
  return `${ratio.toFixed(2)}%`;
}

function formatText(value) {
  const text = value === null || value === undefined ? "" : String(value).trim();
  return text || "--";
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function compactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(
    Number(value),
  );
}

function formatFundamental({ type, value, currency }) {
  switch (type) {
    case "currency":
      return formatCurrency(value, currency);
    case "compactCurrency":
      return compactCurrency(value, currency);
    case "percent":
      return formatPercent(value);
    case "compactNumber":
      return compactNumber(value);
    case "text":
      return formatText(value);
    default:
      return formatNumber(value);
  }
}

function formatMetricValue(key, value, currency = "USD") {
  switch (key) {
    case "profit_margin":
    case "roe":
      return formatPercent(value);
    case "market_cap":
      return compactCurrency(value, currency);
    case "price":
      return formatCurrency(value, currency);
    default:
      return formatNumber(value);
  }
}

function formatNewsTimestamp(value) {
  if (!value) return "";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "";
  return dt.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function CandleTooltip({ active, payload, label, currency }) {
  if (!active || !payload?.length) return null;
  const row = payload[0]?.payload || {};
  const open = row.open ?? row.price;
  const close = row.close ?? row.price;
  const high = row.high ?? Math.max(open ?? 0, close ?? 0);
  const low = row.low ?? Math.min(open ?? 0, close ?? 0);
  const volume = Number(row.volume);
  const volumeLabel = Number.isFinite(volume) ? compactNumber(volume) : null;

  return (
    <Paper
      elevation={3}
      sx={{
        p: 1,
        borderRadius: 1,
        border: "1px solid",
        borderColor: "divider",
        minWidth: 140,
      }}
    >
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2">Open: {formatCurrency(open, currency)}</Typography>
      <Typography variant="body2">High: {formatCurrency(high, currency)}</Typography>
      <Typography variant="body2">Low: {formatCurrency(low, currency)}</Typography>
      <Typography variant="body2">Close: {formatCurrency(close, currency)}</Typography>
      {volumeLabel ? <Typography variant="body2">Volume: {volumeLabel}</Typography> : null}
    </Paper>
  );
}

function CandlestickLayer({ data, yDomain }) {
  const plotArea = usePlotArea();
  if (!plotArea || !data?.length) return null;
  const [min, max] = yDomain;
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return null;

  const step = data.length > 1 ? plotArea.width / (data.length - 1) : plotArea.width;
  const candleWidth = Math.max(2, Math.min(step * 0.6, 14));
  const toY = (value) => plotArea.y + ((max - value) / (max - min)) * plotArea.height;

  return (
    <g>
      {data.map((entry, index) => {
        const open = Number(entry.open ?? entry.price);
        const close = Number(entry.close ?? entry.price);
        const high = Number(entry.high ?? Math.max(open, close));
        const low = Number(entry.low ?? Math.min(open, close));
        if ([open, close, high, low].some((v) => Number.isNaN(v))) return null;

        const x =
          plotArea.x + (data.length > 1 ? index * step : plotArea.width / 2);
        const yOpen = toY(open);
        const yClose = toY(close);
        const yHigh = toY(high);
        const yLow = toY(low);
        const bodyTop = Math.min(yOpen, yClose);
        const bodyHeight = Math.max(1, Math.abs(yOpen - yClose));
        const color = close > open ? "#2e7d32" : close < open ? "#c62828" : "#6b7280";

        return (
          <g key={`${entry.ts ?? entry.label ?? index}`}>
            <line x1={x} x2={x} y1={yHigh} y2={yLow} stroke={color} strokeWidth={1} />
            <rect
              x={x - candleWidth / 2}
              y={bodyTop}
              width={candleWidth}
              height={bodyHeight}
              fill={color}
              stroke={color}
              rx={1}
            />
          </g>
        );
      })}
    </g>
  );
}

function FundamentalsTable({ fundamentals, currency }) {
  const rows = useMemo(
    () =>
      FUNDAMENTAL_FIELDS.map((field) => ({
        ...field,
        value: fundamentals?.[field.key],
      })),
    [fundamentals],
  );

  return (
    <Table size="small">
      <TableBody>
        {rows.map((row) => (
          <TableRow key={row.key}>
            <TableCell sx={{ width: "45%" }}>
              <Typography variant="body2" fontWeight={600}>
                {row.label}
              </Typography>
            </TableCell>
            <TableCell align="right">
              <Typography variant="body2" color="text.primary">
                {formatFundamental({ type: row.type, value: row.value, currency })}
              </Typography>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export default function TickerAnalysisPage() {
  const location = useLocation();
  const searchTicker = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get("ticker")?.trim() || "";
  }, [location.search]);

  const [ticker, setTicker] = useState(searchTicker || "AAPL");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedTf, setSelectedTf] = useState("6M");
  const [chartType, setChartType] = useState("line");
  const [aiInsights, setAiInsights] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState("");
  const [aiQuestion, setAiQuestion] = useState("");
  const [aiQaLoading, setAiQaLoading] = useState(false);
  const [aiQaError, setAiQaError] = useState("");
  const [aiQaResponse, setAiQaResponse] = useState(null);
  const [aiNews, setAiNews] = useState(null);
  const [aiNewsLoading, setAiNewsLoading] = useState(false);
  const [aiNewsError, setAiNewsError] = useState("");
  const aiRequestRef = useRef(0);
  const aiNewsRequestRef = useRef(0);
  const [sectorComparison, setSectorComparison] = useState(null);
  const [sectorLoading, setSectorLoading] = useState(false);
  const [sectorError, setSectorError] = useState("");
  const sectorRequestRef = useRef(0);

  function buildNewsPayload(currentData) {
    const newsList = Array.isArray(currentData?.news) ? currentData.news : [];
    return newsList.slice(0, 12).map((item) => ({
      title: item.title,
      publisher: item.publisher,
      published_at: item.published_at,
      link: item.link,
      summary: item.summary,
    }));
  }

  async function fetchTicker(symbol) {
    const trimmed = symbol.trim();
    if (!trimmed) {
      setError("Enter a ticker to search.");
      setData(null);
      return;
    }

    aiRequestRef.current += 1;
    aiNewsRequestRef.current += 1;
    setAiInsights(null);
    setAiError("");
    setAiLoading(false);
    setAiNews(null);
    setAiNewsError("");
    setAiNewsLoading(false);
    sectorRequestRef.current += 1;
    setSectorComparison(null);
    setSectorError("");
    setSectorLoading(false);
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/api/ticker/overview?ticker=${encodeURIComponent(trimmed)}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const body = await res.json();
      setData(body);
    } catch (e) {
      setError(e.message || String(e));
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  async function fetchAiAnalysis(symbol, nextData, timeframe) {
    const trimmed = (symbol || "").trim();
    if (!trimmed || !nextData?.prices?.[timeframe]?.length) {
      setAiInsights(null);
      setAiLoading(false);
      setAiError("");
      return;
    }

    const reqId = aiRequestRef.current + 1;
    aiRequestRef.current = reqId;
    setAiLoading(true);
    setAiError("");

    try {
      const res = await fetch(`${API_BASE}/api/ticker/ai-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: trimmed,
          timeframe,
          prices: nextData.prices?.[timeframe] || [],
          fundamentals: nextData.fundamentals || {},
          news: buildNewsPayload(nextData),
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        let message = text;
        try {
          const parsed = JSON.parse(text);
          message = parsed.detail || parsed.error || parsed.message || text;
        } catch {
          message = text;
        }
        throw new Error(message || `HTTP ${res.status}`);
      }
      const body = await res.json();
      if (aiRequestRef.current !== reqId) return;
      setAiInsights(body);
    } catch (e) {
      if (aiRequestRef.current !== reqId) return;
      setAiError(e.message || "AI analysis failed.");
      setAiInsights(null);
    } finally {
      if (aiRequestRef.current === reqId) {
        setAiLoading(false);
      }
    }
  }

  async function fetchAiNewsAnalysis(symbol, currentData) {
    const trimmed = (symbol || "").trim();
    const newsPayload = buildNewsPayload(currentData);
    if (!trimmed || !newsPayload.length) {
      setAiNews(null);
      setAiNewsLoading(false);
      setAiNewsError("");
      return;
    }

    const reqId = aiNewsRequestRef.current + 1;
    aiNewsRequestRef.current = reqId;
    setAiNewsLoading(true);
    setAiNewsError("");

    try {
      const res = await fetch(`${API_BASE}/api/ticker/news-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: trimmed,
          news: newsPayload,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        let message = text;
        try {
          const parsed = JSON.parse(text);
          message = parsed.detail || parsed.error || parsed.message || text;
        } catch {
          message = text;
        }
        throw new Error(message || `HTTP ${res.status}`);
      }
      const body = await res.json();
      if (aiNewsRequestRef.current !== reqId) return;
      setAiNews(body);
    } catch (e) {
      if (aiNewsRequestRef.current !== reqId) return;
      setAiNewsError(e.message || "AI news analysis failed.");
      setAiNews(null);
    } finally {
      if (aiNewsRequestRef.current === reqId) {
        setAiNewsLoading(false);
      }
    }
  }

  async function fetchSectorComparison(symbol) {
    const trimmed = (symbol || "").trim();
    if (!trimmed) {
      setSectorComparison(null);
      setSectorError("");
      setSectorLoading(false);
      return;
    }

    const reqId = sectorRequestRef.current + 1;
    sectorRequestRef.current = reqId;
    setSectorLoading(true);
    setSectorError("");

    try {
      const res = await fetch(`${API_BASE}/api/ticker/sector-comparison?ticker=${encodeURIComponent(trimmed)}`);
      if (!res.ok) {
        const text = await res.text();
        let message = text;
        try {
          const parsed = JSON.parse(text);
          message = parsed.detail || parsed.error || parsed.message || text;
        } catch {
          message = text;
        }
        throw new Error(message || `HTTP ${res.status}`);
      }
      const body = await res.json();
      if (sectorRequestRef.current !== reqId) return;
      setSectorComparison(body);
    } catch (e) {
      if (sectorRequestRef.current !== reqId) return;
      setSectorError(e.message || "Sector comparison failed.");
      setSectorComparison(null);
    } finally {
      if (sectorRequestRef.current === reqId) {
        setSectorLoading(false);
      }
    }
  }

  useEffect(() => {
    if (searchTicker) {
      setTicker(searchTicker);
      fetchTicker(searchTicker);
      return;
    }
    fetchTicker(ticker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTicker]);

  const fundamentals = useMemo(
    () => data?.fundamentals || {},
    [data],
  );
  const businessSummary = useMemo(() => {
    const summary =
      fundamentals?.business_summary ||
      fundamentals?.longBusinessSummary ||
      fundamentals?.long_business_summary ||
      fundamentals?.summary ||
      "";
    return typeof summary === "string" ? summary.trim() : String(summary || "").trim();
  }, [fundamentals]);
  const currency = fundamentals?.currency || "USD";
  const priceSeries = useMemo(
    () => data?.prices?.[selectedTf] || [],
    [data, selectedTf],
  );
  const timeframeLabel = TIMEFRAME_OPTIONS.find((opt) => opt.tf === selectedTf)?.label || selectedTf;
  const chartData = useMemo(() => {
    const withOhlc = (row) => {
      const close = row.close ?? row.price;
      const volumeValue = Number(row.volume);
      const open = row.open ?? close;
      const direction = Number.isFinite(open) && Number.isFinite(close)
        ? close > open
          ? "up"
          : close < open
            ? "down"
            : "flat"
        : "flat";
      const volumeColor =
        direction === "up" ? "#2e7d32" : direction === "down" ? "#c62828" : "#6b7280";
      return {
        ...row,
        price: row.price ?? close,
        close,
        open,
        high: row.high ?? close,
        low: row.low ?? close,
        volume: Number.isFinite(volumeValue) ? volumeValue : null,
        volumeColor,
      };
    };
    if (selectedTf !== "1D") {
      return priceSeries.map(withOhlc);
    }
    // For 1D, ensure we sort by ts if present and keep label as HH:MM ET.
    const sorted = [...priceSeries].sort((a, b) => {
      if (a.ts && b.ts) {
        return new Date(a.ts).getTime() - new Date(b.ts).getTime();
      }
      return 0;
    });
    return sorted.map((row) => ({
      ...withOhlc(row),
      timeLabel: row.label,
    }));
  }, [priceSeries, selectedTf]);

  const volumeDomain = useMemo(() => {
    const volumes = chartData
      .map((row) => row.volume)
      .filter((v) => Number.isFinite(v) && v >= 0);
    if (!volumes.length) return null;
    const max = Math.max(...volumes);
    const pad = Math.max(1, max * 0.1);
    return [0, max + pad];
  }, [chartData]);
  const hasVolume = Boolean(volumeDomain);

  const yDomain = chartData.length
    ? chartType === "candle"
      ? (() => {
          const vals = chartData
            .flatMap((p) => [Number(p.low ?? p.price), Number(p.high ?? p.price)])
            .filter((v) => !Number.isNaN(v));
          const min = Math.min(...vals);
          const max = Math.max(...vals);
          const pad = Math.max(0.01, (max - min) * 0.02 || min * 0.002);
          return [min - pad, max + pad];
        })()
      : selectedTf === "1D"
        ? (() => {
            const vals = chartData.map((p) => Number(p.price)).filter((v) => !Number.isNaN(v));
            const min = Math.min(...vals);
            const max = Math.max(...vals);
            const pad = Math.max(0.01, (max - min) * 0.02 || min * 0.002);
            return [min - pad, max + pad];
          })()
        : ["auto", "auto"]
    : ["auto", "auto"];

  const timeframeChange = useMemo(() => {
    if (!chartData?.length) return null;
    const values = chartData
      .map((row) => Number(row.close ?? row.price))
      .filter((v) => Number.isFinite(v));
    if (values.length < 2) return null;
    const start = values[0];
    const end = values[values.length - 1];
    if (start === 0) return null;
    const change = end - start;
    const pct = (change / start) * 100;
    return { start, end, change, pct };
  }, [chartData]);

  const dateRanges = useMemo(() => {
    const ranges = {};
    TIMEFRAME_OPTIONS.forEach(({ tf }) => {
      const series = data?.prices?.[tf] || [];
      if (series.length >= 2) {
        ranges[tf] = { start: series[0].label, end: series[series.length - 1].label };
      }
    });
    return ranges;
  }, [data]);
  const newsItems = useMemo(() => (Array.isArray(data?.news) ? data.news : []), [data]);
  const newsSource = data?.news_source || (newsItems.length ? "yfinance" : "");
  const displayedNews = useMemo(() => newsItems.slice(0, 20), [newsItems]);

  useEffect(() => {
    if (!data) return;
    fetchAiAnalysis(data.ticker || ticker, data, selectedTf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, selectedTf]);

  useEffect(() => {
    if (!data) return;
    fetchAiNewsAnalysis(data.ticker || ticker, data);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  useEffect(() => {
    if (!data?.ticker) return;
    fetchSectorComparison(data.ticker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data?.ticker]);

  useEffect(() => {
    setAiQaResponse(null);
    setAiQaError("");
  }, [data, selectedTf]);

  const intradayTicks = ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"];

  const ratingValue = (aiInsights?.rating || "").toLowerCase();
  const ratingColor =
    ratingValue === "buy" ? "success" : ratingValue === "sell" ? "error" : ratingValue === "hold" ? "warning" : "default";
  const ratingLabel = ratingValue ? ratingValue.charAt(0).toUpperCase() + ratingValue.slice(1) : "No rating";
  const generatedAt = aiInsights?.generated_at ? new Date(aiInsights.generated_at) : null;
  const generatedAtLabel = generatedAt && !Number.isNaN(generatedAt.valueOf()) ? generatedAt.toLocaleString() : null;
  const displayTicker = (data?.ticker || ticker || "").toUpperCase();
  const detailTicker = displayTicker.trim();
  const detailLink = detailTicker ? `/ticker/fundamentals?ticker=${encodeURIComponent(detailTicker)}` : "";
  const displayName = (data?.fundamentals?.name || "").trim();
  const stockTitle = displayName ? `${displayName} (${displayTicker})` : displayTicker;
  const qaGeneratedAt = aiQaResponse?.generated_at ? new Date(aiQaResponse.generated_at) : null;
  const qaGeneratedAtLabel = qaGeneratedAt && !Number.isNaN(qaGeneratedAt.valueOf()) ? qaGeneratedAt.toLocaleString() : null;
  const aiNewsGeneratedAt = aiNews?.generated_at ? new Date(aiNews.generated_at) : null;
  const aiNewsGeneratedLabel =
    aiNewsGeneratedAt && !Number.isNaN(aiNewsGeneratedAt.valueOf())
      ? aiNewsGeneratedAt.toLocaleString()
      : null;
  const sentimentValue = (aiNews?.sentiment || "").toLowerCase();
  const sentimentColor =
    sentimentValue === "bullish"
      ? "success"
      : sentimentValue === "bearish"
        ? "error"
        : sentimentValue
          ? "warning"
          : "default";
  const sentimentLabel = sentimentValue
    ? sentimentValue.charAt(0).toUpperCase() + sentimentValue.slice(1)
    : null;
  const sectorName = (fundamentals?.sector || sectorComparison?.sector || "").trim();
  const sectorScore = typeof sectorComparison?.sector_score === "number" ? sectorComparison.sector_score : null;
  const peersToShow = useMemo(
    () => (Array.isArray(sectorComparison?.peers) ? sectorComparison.peers.slice(0, 6) : []),
    [sectorComparison],
  );

  async function askAiQuestion() {
    const trimmedQuestion = (aiQuestion || "").trim();
    const symbol = (data?.ticker || ticker || "").trim();
    if (!trimmedQuestion) {
      setAiQaError("Enter a question to ask the AI.");
      return;
    }
    if (!symbol) {
      setAiQaError("Load a ticker before asking a question.");
      return;
    }

    setAiQaError("");
    setAiQaResponse(null);
    setAiQaLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/ticker/ai-question`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: symbol,
          timeframe: selectedTf,
          question: trimmedQuestion,
          prices: data?.prices?.[selectedTf] || [],
          fundamentals,
          news: buildNewsPayload(data),
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        let message = text;
        try {
          const parsed = JSON.parse(text);
          message = parsed.detail || parsed.error || parsed.message || text;
        } catch {
          message = text;
        }
        throw new Error(message || `HTTP ${res.status}`);
      }
      const body = await res.json();
      setAiQaResponse(body);
    } catch (e) {
      setAiQaError(e.message || "AI question failed.");
      setAiQaResponse(null);
    } finally {
      setAiQaLoading(false);
    }
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 3 }}>
      <Paper
        sx={{
          p: { xs: 2.5, md: 3 },
          borderRadius: 3,
          background: "linear-gradient(120deg, #0e1f35, #13395e)",
          color: "#fff",
          overflow: "hidden",
          position: "relative",
        }}
        elevation={3}
      >
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background:
              "radial-gradient(circle at 15% 20%, rgba(255,255,255,0.06), transparent 30%), radial-gradient(circle at 85% 10%, rgba(255,255,255,0.05), transparent 24%)",
            pointerEvents: "none",
          }}
        />
        <Stack spacing={2} sx={{ position: "relative" }}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Chip label="Schwab or yfinance" color="secondary" size="small" />
            <Chip label="AI analysis enabled" color="success" size="small" variant="outlined" />
            {data?.price_source && <Chip label={`Prices: ${data.price_source}`} size="small" />}
            {data?.fundamentals_source && (
              <Chip label={`Fundamentals: ${data.fundamentals_source}`} size="small" />
            )}
          </Stack>
          <Typography variant="h5" fontWeight={700}>
            Ticker analysis with multi-timeframe charts + AI takes
          </Typography>
          <Typography variant="body2" sx={{ maxWidth: 800, color: "rgba(255,255,255,0.82)" }}>
            Search any symbol to blend price history, fundamentals, and a quick ChatGPT perspective on the trend and balance
            sheet. Schwab data is used when available with yfinance as a fallback.
          </Typography>
          <Box
            component="form"
            onSubmit={(e) => {
              e.preventDefault();
              fetchTicker(ticker);
            }}
            sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}
          >
            <TextField
              label="Ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              size="small"
              sx={{
                minWidth: 160,
                input: { color: "#fff" },
                label: { color: "rgba(255,255,255,0.82)" },
                "& .MuiOutlinedInput-root": {
                  "& fieldset": { borderColor: "rgba(255,255,255,0.3)" },
                  "&:hover fieldset": { borderColor: "#fff" },
                },
              }}
              InputProps={{
                sx: { color: "#fff" },
              }}
            />
            <Button
              type="submit"
              variant="contained"
              color="secondary"
              startIcon={<Search />}
              disabled={loading}
            >
              {loading ? "Fetching..." : "Search"}
            </Button>
            <Button
              type="button"
              variant="text"
              color="inherit"
              startIcon={<SyncAlt />}
              onClick={() => fetchTicker(ticker)}
              disabled={loading}
          >
            Refresh
          </Button>
        </Box>
          {error && (
            <Alert severity="error" variant="filled">
              {error}
            </Alert>
          )}
        </Stack>
      </Paper>
      {stockTitle ? (
        <Paper
          variant="outlined"
          sx={{
            p: { xs: 2, md: 2.5 },
            display: "flex",
            alignItems: { xs: "flex-start", sm: "center" },
            justifyContent: "space-between",
            gap: 1.5,
            flexWrap: "wrap",
          }}
        >
          <Stack spacing={0.5}>
            <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.6 }}>
              Selected ticker
            </Typography>
            <Typography variant="h6" fontWeight={700}>
              {displayName || displayTicker}
            </Typography>
            {displayName ? (
              <Typography variant="body2" color="text.secondary">
                {displayTicker}
              </Typography>
            ) : null}
            {sectorName ? (
              <Chip label={`Sector: ${sectorName}`} size="small" variant="outlined" color="info" />
            ) : null}
          </Stack>
          <Chip label={displayTicker || "—"} color="primary" size="medium" />
        </Paper>
      ) : null}

      <Paper sx={{ p: 2.5, display: "grid", gap: 1.5 }} variant="outlined">
        <Stack
          direction={{ xs: "column", sm: "row" }}
          justifyContent="space-between"
          alignItems="center"
          gap={1}
        >
          <Stack direction="row" spacing={1} alignItems="center">
            <ShowChart fontSize="small" color="primary" />
            <Typography variant="subtitle1" fontWeight={700}>
              Price History
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {timeframeLabel}
            </Typography>
            {timeframeChange ? (
              <Chip
                size="small"
                color={
                  timeframeChange.pct > 0 ? "success" : timeframeChange.pct < 0 ? "error" : "default"
                }
                variant="outlined"
                label={`${timeframeChange.pct >= 0 ? "+" : ""}${timeframeChange.pct.toFixed(
                  2,
                )}% (${timeframeChange.change >= 0 ? "+" : ""}${formatCurrency(
                  timeframeChange.change,
                  currency,
                )})`}
                title={`Start ${formatCurrency(timeframeChange.start, currency)} → ${formatCurrency(
                  timeframeChange.end,
                  currency,
                )}`}
              />
            ) : (
              <Chip size="small" variant="outlined" label="Δ --" />
            )}
          </Stack>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel id="price-chart-type-label">Chart</InputLabel>
              <Select
                labelId="price-chart-type-label"
                label="Chart"
                value={chartType}
                onChange={(e) => setChartType(e.target.value)}
              >
                {CHART_TYPE_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {TIMEFRAME_OPTIONS.map((opt) => (
              <Button
                key={opt.tf}
                size="small"
                variant={selectedTf === opt.tf ? "contained" : "outlined"}
                onClick={() => setSelectedTf(opt.tf)}
                sx={{ textTransform: "none" }}
              >
                {opt.label}
              </Button>
            ))}
          </Stack>
        </Stack>
        <Divider />
        <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
          <Box sx={{ flex: 3, minHeight: 360, display: "grid", gap: 1.5 }}>
            {loading ? (
              <Box sx={{ height: "100%", display: "grid", placeItems: "center" }}>
                <CircularProgress />
              </Box>
            ) : !chartData || chartData.length === 0 ? (
              <Box sx={{ height: "100%", display: "grid", placeItems: "center" }}>
                <Typography color="text.secondary">No price data.</Typography>
              </Box>
            ) : chartType === "candle" ? (
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart
                  data={chartData}
                  margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
                  syncId="price-volume"
                >
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.4} />
                  <XAxis
                    dataKey={selectedTf === "1D" ? "timeLabel" : "label"}
                    ticks={selectedTf === "1D" ? intradayTicks : undefined}
                    tick={{ fontSize: 12 }}
                    minTickGap={12}
                  />
                  <YAxis
                    yAxisId="price"
                    width={80}
                    tickFormatter={(v) => formatCurrency(v, currency)}
                    domain={yDomain}
                    label={{
                      value: "Price per share",
                      angle: -90,
                      position: "insideLeft",
                    }}
                  />
                  <Tooltip content={<CandleTooltip currency={currency} />} labelFormatter={(label) => label} />
                  <Line
                    dataKey="close"
                    stroke="transparent"
                    dot={false}
                    isAnimationActive={false}
                    yAxisId="price"
                  />
                  <CandlestickLayer data={chartData} yDomain={yDomain} />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart
                  data={chartData}
                  margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
                  syncId="price-volume"
                >
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.4} />
                  <XAxis
                    dataKey={selectedTf === "1D" ? "timeLabel" : "label"}
                    ticks={selectedTf === "1D" ? intradayTicks : undefined}
                    tick={{ fontSize: 12 }}
                    minTickGap={12}
                  />
                  <YAxis
                    yAxisId="price"
                    width={80}
                    tickFormatter={(v) => formatCurrency(v, currency)}
                    domain={yDomain}
                    label={{
                      value: "Price per share",
                      angle: -90,
                      position: "insideLeft",
                    }}
                  />
                  <Tooltip
                    formatter={(value) => [formatCurrency(value, currency), "Price"]}
                    labelFormatter={(label) => label}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#1976d2"
                    strokeWidth={2}
                    dot={false}
                    yAxisId="price"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            )}
            {!loading && chartData && chartData.length > 0 ? (
              hasVolume ? (
                <ResponsiveContainer width="100%" height={120}>
                  <ComposedChart
                    data={chartData}
                    margin={{ top: 0, right: 10, left: -10, bottom: 0 }}
                    syncId="price-volume"
                  >
                    <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.3} />
                    <XAxis
                      dataKey={selectedTf === "1D" ? "timeLabel" : "label"}
                      ticks={selectedTf === "1D" ? intradayTicks : undefined}
                      tick={{ fontSize: 11 }}
                      minTickGap={12}
                    />
                    <YAxis
                      width={70}
                      tickFormatter={(v) => compactNumber(v)}
                      domain={volumeDomain ?? [0, "auto"]}
                      label={{ value: "Volume", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip
                      formatter={(value) => [compactNumber(value), "Volume"]}
                      labelFormatter={(label) => label}
                    />
                    <Bar
                      dataKey="volume"
                      maxBarSize={18}
                      isAnimationActive={false}
                      barCategoryGap="10%"
                    >
                      {chartData.map((entry, idx) => (
                        <Cell key={entry.ts ?? entry.label ?? idx} fill={entry.volumeColor} />
                      ))}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ display: "grid", placeItems: "center", height: 120 }}>
                  <Typography color="text.secondary">Volume unavailable for this timeframe.</Typography>
                </Box>
              )
            ) : null}
          </Box>
          <Box
            sx={{
              flex: 1,
              minHeight: 260,
              border: "1px solid",
              borderColor: "divider",
              borderRadius: 2,
              p: 2,
              bgcolor: "background.paper",
              display: "grid",
              gap: 1,
            }}
          >
            <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle2" fontWeight={700}>
                AI chart read
              </Typography>
              <Chip size="small" variant="outlined" label={timeframeLabel} />
            </Stack>
            {aiLoading ? (
              <Box sx={{ display: "grid", placeItems: "center", py: 2 }}>
                <CircularProgress size={22} />
              </Box>
            ) : aiError ? (
              <Alert severity="warning">{aiError}</Alert>
            ) : aiInsights?.chart_analysis ? (
              <Stack spacing={1}>
                <Typography variant="body2" color="text.secondary">
                  {aiInsights.model ? `Model: ${aiInsights.model}` : "Model: ChatGPT"}
                  {aiInsights.confidence ? ` • Confidence: ${aiInsights.confidence}` : ""}
                </Typography>
                <Typography variant="body1" sx={{ whiteSpace: "pre-line" }}>
                  {aiInsights.chart_analysis}
                </Typography>
                {generatedAtLabel && (
                  <Typography variant="caption" color="text.secondary">
                    Generated {generatedAtLabel}
                  </Typography>
                )}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                AI will summarize the chart once price data loads.
              </Typography>
            )}
            <Divider sx={{ my: 1 }} />
            <Stack spacing={1}>
              <Typography variant="subtitle2" fontWeight={700}>
                Ask this AI about the chart or fundamentals
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Questions use the loaded prices ({timeframeLabel}) and fundamentals as context.
              </Typography>
              <TextField
                label="Ask a question"
                value={aiQuestion}
                onChange={(e) => setAiQuestion(e.target.value)}
                multiline
                minRows={2}
                fullWidth
                size="small"
                placeholder="e.g., What levels look important? How does valuation compare?"
              />
              <Stack direction="row" spacing={1} alignItems="center">
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<Send />}
                  onClick={askAiQuestion}
                  disabled={aiQaLoading || loading}
                >
                  {aiQaLoading ? "Asking..." : "Ask AI"}
                </Button>
                {qaGeneratedAtLabel ? (
                  <Typography variant="caption" color="text.secondary">
                    Last answer {qaGeneratedAtLabel}
                  </Typography>
                ) : null}
              </Stack>
              {aiQaError ? (
                <Alert severity="warning">{aiQaError}</Alert>
              ) : aiQaResponse?.answer ? (
                <Stack spacing={0.5}>
                  <Typography variant="body1" sx={{ whiteSpace: "pre-line" }}>
                    {aiQaResponse.answer}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {aiQaResponse.model ? `Model: ${aiQaResponse.model}` : "Model: ChatGPT"}
                    {aiQaResponse.timeframe ? ` • ${aiQaResponse.timeframe}` : ""}
                    {qaGeneratedAtLabel ? ` • ${qaGeneratedAtLabel}` : ""}
                  </Typography>
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Ask the AI about levels, catalysts, or fundamentals without leaving the chart.
                </Typography>
              )}
            </Stack>
          </Box>
        </Stack>
      </Paper>

      <Paper sx={{ p: 2.5 }} variant="outlined">
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
          <ShowChart fontSize="small" color="primary" />
          <Typography variant="subtitle1" fontWeight={700}>
            Fundamentals & AI grade
          </Typography>
          {fundamentals?.name && (
            <Chip label={fundamentals.name} size="small" sx={{ ml: 1 }} color="primary" />
          )}
        </Stack>
        <Divider sx={{ mb: 2 }} />
        <Stack spacing={2}>
          <Box
            sx={{
              border: "1px solid",
              borderColor: "divider",
              borderRadius: 2,
              p: 2,
              bgcolor: "background.paper",
            }}
          >
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.75 }}>
              <Typography variant="subtitle2" fontWeight={700}>
                Business summary
              </Typography>
              {displayName ? (
                <Chip label={displayName} size="small" variant="outlined" />
              ) : null}
            </Stack>
            {loading ? (
              <Box sx={{ display: "grid", placeItems: "center", py: 2 }}>
                <CircularProgress size={22} />
              </Box>
            ) : businessSummary ? (
              <Typography variant="body1" sx={{ whiteSpace: "pre-line" }}>
                {businessSummary}
              </Typography>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No business summary available yet. Search a ticker to load its description.
              </Typography>
            )}
          </Box>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems="stretch">
            <Box sx={{ flex: 3, minWidth: 0 }}>
              {loading ? (
                <Box sx={{ display: "grid", placeItems: "center", py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : data && Object.keys(fundamentals || {}).length > 0 ? (
                <>
                  <FundamentalsTable fundamentals={fundamentals} currency={currency} />
                  <Box sx={{ mt: 1 }}>
                    {detailLink ? (
                      <Link
                        component={RouterLink}
                        to={detailLink}
                        underline="hover"
                        sx={{ display: "inline-flex", alignItems: "center", gap: 0.5, fontWeight: 600 }}
                      >
                        View detailed fundamentals, balance sheet, and dates
                        <ShowChart sx={{ fontSize: 16 }} />
                      </Link>
                    ) : (
                      <Typography variant="caption" color="text.secondary">
                        Load a ticker to view detailed fundamentals.
                      </Typography>
                    )}
                  </Box>
                </>
              ) : (
                <Alert severity="info">Search for a ticker to load fundamentals.</Alert>
              )}
              <Divider sx={{ my: 1.5 }} />
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems="flex-start">
                <Typography variant="body2" color="text.secondary" sx={{ minWidth: 120 }}>
                  Date ranges
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {TIMEFRAME_OPTIONS.map((opt) => {
                    const range = dateRanges[opt.tf];
                    return (
                      <Chip
                        key={opt.tf}
                        label={
                          range
                            ? `${opt.label}: ${range.start} → ${range.end}`
                            : `${opt.label}: --`
                        }
                        size="small"
                        variant="outlined"
                      />
                    );
                  })}
                </Stack>
              </Stack>
            </Box>
            <Box
              sx={{
                flex: 2,
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 2,
                p: 2,
                bgcolor: "background.paper",
                display: "grid",
                gap: 1,
                minHeight: 220,
              }}
            >
              <Stack direction="row" alignItems="center" spacing={1} justifyContent="space-between">
                <Typography variant="subtitle2" fontWeight={700}>
                  AI fundamentals grade
                </Typography>
                <Chip
                  size="small"
                  color={ratingColor}
                  variant={ratingColor === "default" ? "outlined" : "filled"}
                  label={ratingLabel}
                />
              </Stack>
              {aiLoading ? (
                <Box sx={{ display: "grid", placeItems: "center", py: 2 }}>
                  <CircularProgress size={22} />
                </Box>
              ) : aiError ? (
                <Alert severity="warning">{aiError}</Alert>
              ) : aiInsights && (aiInsights.fundamental_analysis || aiInsights.rating_reason || ratingValue) ? (
                <Stack spacing={1}>
                  {aiInsights.fundamental_analysis && (
                    <Typography variant="body1" sx={{ whiteSpace: "pre-line" }}>
                      {aiInsights.fundamental_analysis}
                    </Typography>
                  )}
                  {aiInsights.rating_reason && (
                    <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-line" }}>
                      {aiInsights.rating_reason}
                    </Typography>
                  )}
                  {ratingValue && !aiInsights.rating_reason && (
                    <Typography variant="body2" color="text.secondary">
                      Rating call: {ratingLabel}
                    </Typography>
                  )}
                  {aiInsights.model && (
                    <Typography variant="caption" color="text.secondary">
                      Model: {aiInsights.model}
                    </Typography>
                  )}
                  {generatedAtLabel && (
                    <Typography variant="caption" color="text.secondary">
                      Generated {generatedAtLabel}
                    </Typography>
                  )}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  AI will grade the fundamentals after data loads.
                </Typography>
              )}
            </Box>
            <Box
              sx={{
                flex: 2,
                border: "1px solid",
                borderColor: "divider",
                borderRadius: 2,
                p: 2,
                bgcolor: "background.paper",
                display: "grid",
                gap: 1,
                minHeight: 220,
              }}
            >
              <Stack direction="row" alignItems="center" spacing={1} justifyContent="space-between" flexWrap="wrap">
                <Typography variant="subtitle2" fontWeight={700}>
                  Sector context
                </Typography>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                  {sectorName ? <Chip size="small" label={sectorName} color="info" variant="outlined" /> : null}
                  {sectorScore !== null ? (
                    <Chip
                      size="small"
                      color={sectorScore >= 60 ? "success" : sectorScore >= 40 ? "warning" : "default"}
                      variant={sectorScore >= 40 ? "filled" : "outlined"}
                      label={`Sector score: ${Math.round(sectorScore)}/100`}
                    />
                  ) : null}
                </Stack>
              </Stack>
              {sectorLoading ? (
                <Box sx={{ display: "grid", placeItems: "center", py: 2 }}>
                  <CircularProgress size={22} />
                </Box>
              ) : sectorError ? (
                <Alert severity="warning">{sectorError}</Alert>
              ) : sectorComparison ? (
                <Stack spacing={1}>
                  <Typography variant="body2" color="text.secondary">
                    {sectorComparison.peer_count
                      ? `Compared with ${sectorComparison.peer_count} peers.`
                      : "Peer sample building..."}
                  </Typography>
                  <Stack spacing={0.75}>
                    {Object.entries(SECTOR_METRIC_LABELS).map(([key, label]) => {
                      const metric = sectorComparison.metrics?.[key];
                      if (!metric) return null;
                      return (
                        <Stack key={key} spacing={0.25}>
                          <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                            <Typography variant="body2" fontWeight={600}>
                              {label}
                            </Typography>
                            {metric.percentile !== null && metric.percentile !== undefined ? (
                              <Chip
                                size="small"
                                label={`${Math.round(metric.percentile)}th pctile`}
                                color={metric.percentile >= 70 ? "success" : metric.percentile >= 40 ? "warning" : "default"}
                                variant={metric.percentile >= 70 ? "filled" : "outlined"}
                              />
                            ) : null}
                          </Stack>
                          <Typography variant="caption" color="text.secondary">
                            You: {formatMetricValue(key, metric.value, currency)} • Peer median:{" "}
                            {formatMetricValue(key, metric.peer_median, currency)}
                          </Typography>
                        </Stack>
                      );
                    })}
                  </Stack>
                  {peersToShow.length ? (
                    <Table size="small" sx={{ mt: 0.5 }}>
                      <TableHead>
                        <TableRow>
                          <TableCell>Peer</TableCell>
                          <TableCell align="right">P/E</TableCell>
                          <TableCell align="right">Fwd P/E</TableCell>
                          <TableCell align="right">Margin</TableCell>
                          <TableCell align="right">Mkt Cap</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {peersToShow.map((peer) => (
                          <TableRow key={peer.ticker}>
                            <TableCell>
                              <Stack spacing={0.25}>
                                <Typography variant="body2" fontWeight={700}>
                                  {peer.ticker}
                                </Typography>
                                {peer.name ? (
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ maxWidth: 180, display: "block" }}
                                    noWrap
                                  >
                                    {peer.name}
                                  </Typography>
                                ) : null}
                              </Stack>
                            </TableCell>
                            <TableCell align="right">{formatNumber(peer.pe_ratio)}</TableCell>
                            <TableCell align="right">{formatNumber(peer.forward_pe)}</TableCell>
                            <TableCell align="right">{formatPercent(peer.profit_margin)}</TableCell>
                            <TableCell align="right">{compactCurrency(peer.market_cap, currency)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      {sectorComparison.sector
                        ? "Same-sector peers will appear once data loads."
                        : "Sector data unavailable for this ticker."}
                    </Typography>
                  )}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Load a ticker to compare it against its sector peers.
                </Typography>
              )}
            </Box>
          </Stack>
        </Stack>
      </Paper>

      <Paper sx={{ p: 2.5, display: "grid", gap: 1.25 }} variant="outlined">
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
          <Article fontSize="small" color="primary" />
          <Typography variant="subtitle1" fontWeight={700}>
            Recent headlines (yfinance)
          </Typography>
          {newsSource ? <Chip size="small" label={`Source: ${newsSource}`} /> : null}
          {displayedNews.length ? (
            <Chip size="small" variant="outlined" label={`Showing ${displayedNews.length}`} />
          ) : null}
        </Stack>
        <Divider />
        <Box
          sx={{
            border: "1px solid",
            borderColor: "divider",
            borderRadius: 2,
            p: 2,
            bgcolor: "background.paper",
            display: "grid",
            gap: 0.75,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" flexWrap="wrap">
            <Typography variant="subtitle2" fontWeight={700}>
              AI read of headlines
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
              {sentimentLabel ? (
                <Chip
                  size="small"
                  color={sentimentColor}
                  variant={sentimentColor === "default" ? "outlined" : "filled"}
                  label={sentimentLabel}
                />
              ) : null}
              {aiNewsGeneratedLabel ? (
                <Typography variant="caption" color="text.secondary">
                  Updated {aiNewsGeneratedLabel}
                </Typography>
              ) : null}
            </Stack>
          </Stack>
          {aiNewsLoading ? (
            <Box sx={{ display: "grid", placeItems: "center", py: 1 }}>
              <CircularProgress size={22} />
            </Box>
          ) : aiNewsError ? (
            <Alert severity="warning">{aiNewsError}</Alert>
          ) : aiNews?.summary ? (
            <Stack spacing={0.75}>
              <Typography variant="body1" sx={{ whiteSpace: "pre-line" }}>
                {aiNews.summary}
              </Typography>
              {aiNews.themes?.length ? (
                <Stack spacing={0.35}>
                  <Typography variant="caption" color="text.secondary">
                    Themes
                  </Typography>
                  <Stack direction="row" spacing={0.5} flexWrap="wrap">
                    {aiNews.themes.map((theme, idx) => (
                      <Chip key={`${theme}-${idx}`} size="small" label={theme} />
                    ))}
                  </Stack>
                </Stack>
              ) : null}
              {aiNews.risks?.length ? (
                <Stack spacing={0.35}>
                  <Typography variant="caption" color="text.secondary">
                    Watchouts
                  </Typography>
                  <Stack direction="row" spacing={0.5} flexWrap="wrap">
                    {aiNews.risks.map((risk, idx) => (
                      <Chip key={`${risk}-${idx}`} size="small" variant="outlined" color="warning" label={risk} />
                    ))}
                  </Stack>
                </Stack>
              ) : null}
            </Stack>
          ) : displayedNews.length ? (
            <Typography variant="body2" color="text.secondary">
              Headlines loaded; AI summary will appear in a moment.
            </Typography>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No headlines yet for AI to summarize.
            </Typography>
          )}
        </Box>
        {loading ? (
          <Box sx={{ display: "grid", placeItems: "center", py: 3 }}>
            <CircularProgress />
          </Box>
        ) : displayedNews.length ? (
          <Stack spacing={1.25}>
            {displayedNews.map((item, idx) => (
              <Box
                key={item.id || item.link || `${item.title}-${idx}`}
                sx={{ display: "grid", gap: 0.35 }}
              >
                <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap">
                  {item.link ? (
                    <Link
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      underline="hover"
                      sx={{ fontWeight: 700 }}
                    >
                      {item.title}
                      <OpenInNew sx={{ fontSize: 16, ml: 0.5, verticalAlign: "text-bottom" }} />
                    </Link>
                  ) : (
                    <Typography variant="subtitle2" fontWeight={700}>
                      {item.title}
                    </Typography>
                  )}
                  {item.publisher ? <Chip size="small" variant="outlined" label={item.publisher} /> : null}
                </Stack>
                <Typography variant="caption" color="text.secondary">
                  {formatNewsTimestamp(item.published_at) || "Time unknown"}
                </Typography>
                {item.summary ? (
                  <Typography variant="body2" color="text.primary">
                    {item.summary}
                  </Typography>
                ) : null}
                {idx < displayedNews.length - 1 && <Divider sx={{ mt: 1 }} />}
              </Box>
            ))}
          </Stack>
        ) : (
          <Alert severity="info">No headlines yet. Search a ticker to load its news.</Alert>
        )}
      </Paper>
    </Container>
  );
}
