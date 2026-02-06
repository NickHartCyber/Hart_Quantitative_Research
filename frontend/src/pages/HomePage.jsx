import React, { useEffect, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { Article, Insights, TrendingDown, TrendingUp, Timeline } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { alpha, useTheme } from "@mui/material/styles";

const API_BASE = process.env.REACT_APP_API_BASE || "";
const HOME_CACHE_KEY = "mg_home_indices_v1";
const NY_TIMEZONE = "America/New_York";

const readCachedHomeIndices = () => {
  try {
    if (typeof window === "undefined") return null;
    const raw = window.localStorage.getItem(HOME_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.indices)) return null;
    return parsed;
  } catch {
    return null;
  }
};

const writeCachedHomeIndices = (payload) => {
  try {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(HOME_CACHE_KEY, JSON.stringify(payload));
  } catch {
    // ignore storage failures
  }
};

const sentimentColor = (sentiment) => {
  const s = (sentiment || "").toLowerCase();
  if (s.includes("bull")) return "success";
  if (s.includes("bear")) return "error";
  if (s.includes("warn") || s.includes("neutral")) return "warning";
  return "default";
};

const sentimentLabel = (sentiment) => {
  const s = (sentiment || "").toLowerCase();
  if (s.includes("bull")) return "Bullish";
  if (s.includes("bear")) return "Bearish";
  if (s.includes("warn") || s.includes("neutral")) return "Neutral";
  return sentiment || "Neutral";
};

const HEADLINE_TYPE_LABELS = {
  upgrade: "Upgrade",
  downgrade: "Downgrade",
  initiation: "Initiation",
  price_target: "Price Target",
  rating_maintained: "Rating Maintained",
};

const resolveHeadlineBadge = (item) => {
  if (!item) return null;
  const direct = (item.headline_type || "").toLowerCase();
  let key = direct;
  if (!key && Array.isArray(item.tags)) {
    key = item.tags.map((tag) => String(tag).toLowerCase()).find((tag) => HEADLINE_TYPE_LABELS[tag]) || "";
  }
  if (!key) return null;
  return HEADLINE_TYPE_LABELS[key] || key;
};


const parseNumber = (val) => {
  if (val === null || val === undefined) return null;
  const num = typeof val === "number" ? val : Number(val);
  return Number.isFinite(num) ? num : null;
};

const formatPct = (val, digits = 2) => {
  const num = parseNumber(val);
  if (num === null) return "—";
  const fixed = num.toFixed(digits);
  return `${num > 0 ? "+" : ""}${fixed}%`;
};

const formatPrice = (val) => {
  const num = parseNumber(val);
  if (num === null) return "—";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: num >= 100 ? 1 : 2 }).format(num);
};

const formatDeltaUsd = (val) => {
  const num = parseNumber(val);
  if (num === null) return "—";
  const abs = Math.abs(num);
  const formatted = new Intl.NumberFormat("en-US", { maximumFractionDigits: abs >= 100 ? 1 : 2 }).format(abs);
  if (num > 0) return `+$${formatted}`;
  if (num < 0) return `-$${formatted}`;
  return `$${formatted}`;
};

const nyDateKey = (date) => {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: NY_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const map = {};
  for (const part of parts) {
    if (part.type !== "literal") map[part.type] = part.value;
  }
  return map.year && map.month && map.day ? `${map.year}-${map.month}-${map.day}` : null;
};

const getLatestPriceSnapshot = (index, analysisTf) => {
  const directPrice = parseNumber(index?.latest?.price);
  const directTs = index?.latest?.as_of || index?.latest?.ts;
  if (directPrice !== null) {
    const ts = directTs ? new Date(directTs) : null;
    return { price: directPrice, ts };
  }
  const series =
    index?.prices?.["1D"] ||
    index?.prices?.[analysisTf] ||
    index?.prices?.["5D"] ||
    [];
  for (let i = series.length - 1; i >= 0; i -= 1) {
    const row = series[i];
    const price = parseNumber(row?.close ?? row?.price);
    if (price === null) continue;
    const ts = row?.ts ? new Date(row.ts) : null;
    return { price, ts };
  }
  return { price: null, ts: null };
};

const getDailyCloses = (rows = []) => {
  const byDate = new Map();
  for (const row of rows) {
    const ts = row?.ts;
    if (!ts) continue;
    const date = new Date(ts);
    const key = nyDateKey(date);
    if (!key) continue;
    const close = parseNumber(row?.close ?? row?.price);
    if (close === null) continue;
    byDate.set(key, close);
  }
  return Array.from(byDate.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([dateKey, close]) => ({ dateKey, close }));
};

const resolvePreviousClose = (index, latestDateKey, todayKey) => {
  const dailyRows = index?.prices?.["1M"] || index?.prices?.["5D"] || [];
  const closes = getDailyCloses(dailyRows);
  if (!closes.length) return null;
  if (latestDateKey && todayKey && latestDateKey === todayKey) {
    for (let i = closes.length - 1; i >= 0; i -= 1) {
      if (closes[i].dateKey < todayKey) return closes[i].close;
    }
  }
  return closes[closes.length - 1].close;
};

const getDayMove = (index, analysisTf) => {
  const { price, ts } = getLatestPriceSnapshot(index, analysisTf);
  const latestDateKey = ts ? nyDateKey(ts) : null;
  const todayKey = nyDateKey(new Date());
  const prevClose = resolvePreviousClose(index, latestDateKey, todayKey);
  if (price === null || prevClose === null) {
    return { pct: null, delta: null, prevClose };
  }
  const delta = price - prevClose;
  const pct = prevClose ? (delta / prevClose) * 100 : null;
  return { pct, delta, prevClose };
};

const changeIcon = (val) => {
  const num = parseNumber(val) || 0;
  return num < 0 ? <TrendingDown fontSize="small" color="error" /> : <TrendingUp fontSize="small" color="success" />;
};

const statPct = (index, tf) => formatPct(index?.stats?.[tf]?.pct_change);

const latestPrice = (index, tf = "6M") => {
  const value =
    index?.latest?.price ??
    index?.stats?.["1D"]?.end ??
    index?.stats?.["5D"]?.end ??
    index?.stats?.[tf]?.end ??
    null;
  return formatPrice(value);
};

const topNews = (news = [], limit = 3) => (Array.isArray(news) ? news.slice(0, limit) : []);

const aiSummary = (index, loading) => {
  if (loading && !index?.ai) return "Loading AI view…";
  if (index?.ai?.chart_analysis) return index.ai.chart_analysis;
  if (index?.ai?.summary) return index.ai.summary;
  if (index?.ai_error) return index.ai_error;
  return "AI summary unavailable.";
};

export default function HomePage() {
  const navigate = useNavigate();
  const theme = useTheme();
  const [initialCache] = useState(() => readCachedHomeIndices());
  const [indices, setIndices] = useState(() =>
    initialCache?.indices && Array.isArray(initialCache.indices) ? initialCache.indices : [],
  );
  const [analysisTf, setAnalysisTf] = useState(() => initialCache?.analysisTf || initialCache?.analysis_timeframe || "6M");
  const [asOf, setAsOf] = useState(() => initialCache?.asOf || initialCache?.as_of || "");
  const [loading, setLoading] = useState(!initialCache);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const fetchIndices = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/market/indices?cache_only=true`);
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`);
        }
        const data = await res.json();
        if (cancelled) return;
        const nextIndices = Array.isArray(data.indices) ? data.indices : [];
        const nextAnalysisTf = data.analysis_timeframe || data.analysisTf || "6M";
        const nextAsOf = data.as_of || data.asOf || "";
        setIndices(nextIndices);
        setAnalysisTf(nextAnalysisTf);
        setAsOf(nextAsOf);
        setError("");
        writeCachedHomeIndices({
          indices: nextIndices,
          analysisTf: nextAnalysisTf,
          asOf: nextAsOf,
          fetchedAt: Date.now(),
        });
      } catch (err) {
        if (cancelled) return;
        const message = err?.message || "Unable to load cached market data.";
        setError(initialCache ? `${message} Showing last cached snapshot.` : message);
        if (!initialCache) {
          setIndices([]);
          setAnalysisTf("6M");
          setAsOf("");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchIndices();
    return () => {
      cancelled = true;
    };
  }, [initialCache]);

  const hasData = indices.length > 0;

  return (
    <Container maxWidth="lg" sx={{ py: 5, display: "grid", gap: 3 }}>
      {error && (
        <Alert severity="warning" variant="outlined">
          Cached market data unavailable. {error}
        </Alert>
      )}

      {/* Hero */}
      <Paper
        sx={{
          p: { xs: 3, md: 4 },
          borderRadius: 3,
          background: `linear-gradient(135deg, ${theme.palette.primary.dark}, #0b203f 55%, ${theme.palette.secondary.main})`,
          color: "#fff",
          position: "relative",
          overflow: "hidden",
          border: `1px solid ${alpha(theme.palette.primary.light, 0.3)}`,
          boxShadow: "0 30px 80px rgba(8, 22, 44, 0.45)",
        }}
        elevation={0}
      >
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background: `radial-gradient(circle at 20% 20%, ${alpha(
              "#9bb6e0",
              0.18
            )}, transparent 32%), radial-gradient(circle at 80% 0%, ${alpha("#6fb6ff", 0.18)}, transparent 26%)`,
            pointerEvents: "none",
          }}
        />
        <Stack spacing={2} sx={{ position: "relative" }}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Chip label="Market pulse" color="secondary" variant="filled" sx={{ color: "white" }} />
            <Chip label="AI bull vs bear read" variant="outlined" sx={{ borderColor: "rgba(255,255,255,0.4)", color: "white" }} />
            <Chip label={`Trend horizon: ${analysisTf}`} variant="outlined" sx={{ borderColor: "rgba(255,255,255,0.25)", color: "white" }} />
            {asOf && (
              <Chip
                label={`As of ${new Date(asOf).toLocaleString()}`}
                size="small"
                sx={{ borderColor: "rgba(255,255,255,0.25)", color: "white" }}
                variant="outlined"
              />
            )}
            {loading && (
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={16} sx={{ color: "white" }} />
                <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.8)" }}>
                  Loading cached data…
                </Typography>
              </Stack>
            )}
          </Stack>
          <Typography variant="h4" fontWeight={700}>
            General market trend analysis using ETF proxies (SPY, DIA, QQQ, IWM)
          </Typography>
          <Typography variant="body1" sx={{ maxWidth: 820, color: "rgba(255,255,255,0.82)" }}>
            Cached direction, trend strength, and AI commentary on bull vs bear posture, with news for each ETF proxy.
          </Typography>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
            <Button variant="contained" color="secondary" onClick={() => navigate("/suggester")}>
              View Stock Ideas
            </Button>
            <Button variant="outlined" color="inherit" onClick={() => navigate("/ticker")}>
              Run a ticker check
            </Button>
          </Stack>
          {hasData ? (
            <Grid container spacing={1} sx={{ mt: 1 }}>
              {indices.map((index) => {
                const dayMove = getDayMove(index, analysisTf);
                const change = dayMove.pct ?? index?.stats?.["1D"]?.pct_change ?? index?.stats?.[analysisTf]?.pct_change;
                const sentiment = index?.sentiment || "neutral";
                return (
                  <Grid item xs={6} sm={3} key={index.key || index.name}>
                    <Paper
                      sx={{
                        p: 1.5,
                        bgcolor: "rgba(255,255,255,0.08)",
                        borderColor: "rgba(255,255,255,0.18)",
                        color: "white",
                        backdropFilter: "blur(6px)",
                      }}
                      variant="outlined"
                    >
                      <Stack spacing={0.5}>
                        <Stack direction="row" alignItems="center" spacing={0.75}>
                          {changeIcon(change)}
                          <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.8)" }}>
                            {index.name}
                          </Typography>
                        </Stack>
                        <Typography variant="h6">{formatPct(change)}</Typography>
                        <Typography variant="caption" sx={{ color: "rgba(255,255,255,0.7)" }}>
                          {formatDeltaUsd(dayMove.delta)}
                        </Typography>
                        <Chip
                          label={sentimentLabel(sentiment)}
                          size="small"
                          color={sentimentColor(sentiment)}
                          variant="outlined"
                          sx={{ borderColor: "rgba(255,255,255,0.3)", color: "white", width: "fit-content" }}
                        />
                      </Stack>
                    </Paper>
                  </Grid>
                );
              })}
            </Grid>
          ) : (
            !loading && (
              <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.85)", mt: 1 }}>
                No cached ETF proxy data available right now.
              </Typography>
            )
          )}
        </Stack>
      </Paper>

      {/* Trend breakdown */}
      <Stack spacing={1}>
        <Typography variant="h6">Trendboard for the ETF proxies</Typography>
        <Typography variant="body2" color="text.secondary">
          Levels, recent momentum, and an AI-annotated bias for each ETF proxy.
        </Typography>
      </Stack>
      {hasData ? (
        <Grid container spacing={2}>
          {indices.map((index) => {
            const dayMove = getDayMove(index, analysisTf);
            const dayPct = dayMove.pct ?? index?.stats?.["1D"]?.pct_change;
            return (
              <Grid item xs={12} md={6} key={index.key || index.name}>
                <Paper
                  sx={{
                    p: 2.5,
                    height: "100%",
                    display: "grid",
                    gap: 1.5,
                    background: "linear-gradient(180deg, #ffffff, #f7f9fd)",
                    borderColor: alpha(theme.palette.primary.main, 0.08),
                  }}
                  variant="outlined"
                >
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Typography variant="subtitle1" fontWeight={700}>
                        {index.name}
                      </Typography>
                      <Chip label={index.ticker} size="small" variant="outlined" />
                    </Stack>
                    <Chip label={sentimentLabel(index.sentiment)} color={sentimentColor(index.sentiment)} size="small" />
                  </Stack>

                  <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                    {changeIcon(dayPct)}
                    <Typography variant="h4" component="div">
                      {latestPrice(index, analysisTf)}
                    </Typography>
                    <Chip label={`${formatPct(dayPct)} 1D`} color={parseNumber(dayPct) < 0 ? "error" : "success"} size="small" />
                    <Chip label={formatDeltaUsd(dayMove.delta)} variant="outlined" size="small" />
                    <Chip label={`${statPct(index, "5D")} 5D`} variant="outlined" size="small" />
                    <Chip label={`${statPct(index, analysisTf)} ${analysisTf}`} variant="outlined" size="small" />
                  </Stack>

                  <Typography variant="body2" color="text.secondary">
                    Source: {index.price_source || "tiingo"} · Headlines: {index.news_source || "rss"} {index.ai_error ? `· AI: ${index.ai_error}` : ""}
                  </Typography>

                  <Divider />
                  <Stack spacing={1}>
                    <Typography variant="subtitle2" sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                      <Insights fontSize="small" color="primary" />
                      AI take
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {aiSummary(index, loading)}
                    </Typography>
                  </Stack>
                </Paper>
              </Grid>
            );
          })}
        </Grid>
      ) : (
        <Paper variant="outlined" sx={{ p: 2.5, background: "linear-gradient(180deg, #ffffff, #f7f9fd)" }}>
          {loading ? (
            <Stack direction="row" spacing={1} alignItems="center">
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Loading cached market data…
              </Typography>
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No cached market data available right now.
            </Typography>
          )}
        </Paper>
      )}

      {/* News */}
      <Paper
        sx={{
          p: { xs: 2.5, md: 3 },
          display: "grid",
          gap: 2,
          background: "linear-gradient(180deg, #ffffff, #f7f9fc)",
          borderColor: alpha(theme.palette.primary.main, 0.08),
        }}
        variant="outlined"
      >
        <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ xs: "flex-start", md: "center" }}>
          <Box>
            <Typography variant="h6">News by ETF proxy</Typography>
            <Typography variant="body2" color="text.secondary">
              Headlines tied to each of the four ETF proxies to keep the context close.
            </Typography>
          </Box>
          <Chip icon={<Article fontSize="small" />} label="Fresh tape headlines" variant="outlined" />
        </Stack>
        {hasData ? (
          <Grid container spacing={2}>
            {indices.map((index) => (
              <Grid item xs={12} md={3} key={index.key || index.name}>
                <Paper sx={{ p: 2, height: "100%", display: "grid", gap: 1 }} variant="outlined">
                  <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                    <Typography variant="subtitle1" fontWeight={700}>
                      {index.name}
                    </Typography>
                    <Chip label={sentimentLabel(index.sentiment)} color={sentimentColor(index.sentiment)} size="small" />
                  </Stack>
                  <List dense disablePadding>
                    {topNews(index.news).length === 0 && (
                      <ListItem disableGutters>
                        <ListItemText
                          primaryTypographyProps={{ variant: "body2", color: "text.secondary" }}
                          primary="No recent headlines yet."
                        />
                      </ListItem>
                    )}
                    {topNews(index.news).map((item) => {
                      const link = item.link || item.url;
                      const badge = resolveHeadlineBadge(item);
                      const secondary = [item.publisher || item.source || item.published_at || "", badge].filter(Boolean).join(" · ");
                      const primary = (
                        <Typography
                          variant="body2"
                          fontWeight={600}
                          component={link ? "a" : "span"}
                          href={link || undefined}
                          target={link ? "_blank" : undefined}
                          rel={link ? "noopener noreferrer" : undefined}
                          sx={{ color: "text.primary", textDecoration: link ? "none" : "inherit", "&:hover": link ? { textDecoration: "underline" } : undefined }}
                        >
                          {item.title}
                        </Typography>
                      );
                      return (
                        <ListItem key={`${index.key || index.name}-${item.title}`} disableGutters alignItems="flex-start">
                          <ListItemIcon sx={{ minWidth: 30 }}>
                            <Timeline fontSize="small" color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary={primary}
                            secondaryTypographyProps={{ variant: "caption", color: "text.secondary" }}
                            secondary={secondary}
                          />
                        </ListItem>
                      );
                    })}
                  </List>
                </Paper>
              </Grid>
            ))}
          </Grid>
        ) : (
          <Typography variant="body2" color="text.secondary">
            {loading ? "Loading headlines…" : "No headlines available without cached ETF proxy data."}
          </Typography>
        )}
      </Paper>
    </Container>
  );
}
