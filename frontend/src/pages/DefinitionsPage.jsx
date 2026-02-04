import React, { useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  Container,
  Divider,
  Grid,
  InputAdornment,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { Info, Lightbulb, MenuBook, Search } from "@mui/icons-material";
import { financialTerms } from "../data/financialTerms";

function TermCard({ item }) {
  return (
    <Paper variant="outlined" sx={{ p: 2.25, height: "100%", display: "grid", gap: 1.25 }}>
      <Stack direction="row" spacing={1.5} alignItems="center" justifyContent="space-between">
        <Box>
          <Typography variant="subtitle1" fontWeight={700}>
            {item.term}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {item.category}
          </Typography>
        </Box>
        <Chip label={item.category} size="small" color="primary" variant="outlined" />
      </Stack>

      <Typography variant="body2" color="text.primary">
        {item.definition}
      </Typography>

      {item.whyItMatters && (
        <Stack direction="row" spacing={1} alignItems="flex-start" color="text.secondary">
          <Lightbulb fontSize="small" color="warning" />
          <Typography variant="body2">
            <strong>Why it matters: </strong>
            {item.whyItMatters}
          </Typography>
        </Stack>
      )}

      {item.example && (
        <Box
          sx={{
            p: 1.25,
            borderRadius: 1.25,
            bgcolor: (theme) => theme.palette.action.hover,
          }}
        >
          <Typography variant="body2">
            <strong>Example: </strong>
            {item.example}
          </Typography>
        </Box>
      )}

      {item.tags?.length ? (
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
          {item.tags.map((tag) => (
            <Chip key={tag} size="small" label={tag} variant="outlined" />
          ))}
        </Stack>
      ) : null}
    </Paper>
  );
}

export default function DefinitionsPage() {
  const [query, setQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");

  const categories = useMemo(
    () => ["All", ...Array.from(new Set(financialTerms.map((item) => item.category)))],
    [],
  );

  const filteredTerms = useMemo(() => {
    const needle = query.trim().toLowerCase();

    return financialTerms
      .filter((item) => selectedCategory === "All" || item.category === selectedCategory)
      .filter((item) => {
        if (!needle) return true;
        const haystack = [
          item.term,
          item.definition,
          item.whyItMatters,
          item.example,
          item.category,
          ...(item.tags || []),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return haystack.includes(needle);
      })
      .sort((a, b) => a.term.localeCompare(b.term));
  }, [query, selectedCategory]);

  return (
    <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 3 }}>
      <Paper
        elevation={3}
        sx={{
          p: { xs: 2.5, md: 3 },
          borderRadius: 3,
          background: "linear-gradient(120deg, #0f2033, #123f63 50%, #0f2033)",
          color: "white",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background:
              "radial-gradient(circle at 20% 30%, rgba(255,255,255,0.08), transparent 28%), radial-gradient(circle at 80% 0%, rgba(255,255,255,0.05), transparent 22%)",
            pointerEvents: "none",
          }}
        />
        <Stack spacing={2} sx={{ position: "relative" }}>
          <Stack direction="row" spacing={1} alignItems="center">
            <MenuBook fontSize="small" />
            <Typography variant="overline" sx={{ letterSpacing: 1, color: "rgba(255,255,255,0.78)" }}>
              Financial definitions
            </Typography>
          </Stack>
          <Typography variant="h5" fontWeight={700}>
            Searchable glossary for trading, risk, and fundamentals
          </Typography>
          <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.85)", maxWidth: 960 }}>
            Browse concise explanations of the metrics and jargon used across Hart Quantitative Research. Filter by topic or type a
            term to jump straight to what you need.
          </Typography>

          <Stack direction={{ xs: "column", md: "row" }} spacing={1.5} alignItems={{ xs: "stretch", md: "center" }}>
            <TextField
              fullWidth
              placeholder="Search by term, tag, or concept (e.g., 'volatility', 'margin', 'breakout')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search sx={{ color: "rgba(255,255,255,0.85)" }} />
                  </InputAdornment>
                ),
              }}
              sx={{
                "& .MuiOutlinedInput-root": {
                  backgroundColor: "rgba(0,0,0,0.22)",
                  color: "white",
                  borderRadius: 2,
                  "& fieldset": { borderColor: "rgba(255,255,255,0.25)" },
                  "&:hover fieldset": { borderColor: "rgba(255,255,255,0.45)" },
                  "&.Mui-focused fieldset": { borderColor: "#7dc2ff" },
                },
                "& .MuiInputBase-input::placeholder": { color: "rgba(255,255,255,0.75)" },
              }}
            />
            <Button
              variant="outlined"
              color="inherit"
              onClick={() => {
                setQuery("");
                setSelectedCategory("All");
              }}
              sx={{ borderColor: "rgba(255,255,255,0.5)", color: "white", width: { xs: "100%", md: "auto" } }}
            >
              Reset filters
            </Button>
          </Stack>

          <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems={{ xs: "flex-start", sm: "center" }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Info fontSize="small" />
              <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.8)" }}>
                {filteredTerms.length} of {financialTerms.length} terms visible
              </Typography>
            </Stack>
            <Divider flexItem orientation="vertical" sx={{ borderColor: "rgba(255,255,255,0.2)", display: { xs: "none", sm: "block" } }} />
            <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.72)" }}>
              Categories: {categories.length - 1}
            </Typography>
          </Stack>
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2.25, borderRadius: 2 }}>
        <Stack spacing={1.5}>
          <Typography variant="subtitle1" fontWeight={700}>
            Filter by topic
          </Typography>
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
            {categories.map((cat) => {
              const selected = selectedCategory === cat;
              return (
                <Chip
                  key={cat}
                  label={cat}
                  color={selected ? "primary" : "default"}
                  variant={selected ? "filled" : "outlined"}
                  onClick={() => setSelectedCategory(cat)}
                  clickable
                />
              );
            })}
          </Stack>
        </Stack>
      </Paper>

      {filteredTerms.length === 0 ? (
        <Alert severity="info">No matching terms yet. Try a broader search or clear the filters.</Alert>
      ) : (
        <Grid container spacing={2}>
          {filteredTerms.map((item) => (
            <Grid item xs={12} md={6} key={item.term}>
              <TermCard item={item} />
            </Grid>
          ))}
        </Grid>
      )}
    </Container>
  );
}
