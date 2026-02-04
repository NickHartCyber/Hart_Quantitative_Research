import React from "react";
import { Container, Paper, Stack, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import Link from "@mui/material/Link";

const PARAGRAPHS = [
  "The content provided on this website, including but not limited to stock ideas, market signals, trade candidates, analysis, charts, and commentary, is provided for educational and informational purposes only.",
  "Hart Quantitative Research is not a registered investment advisor, broker-dealer, or financial advisor. No content on this website should be construed as personalized investment advice or a recommendation to buy, sell, or hold any security.",
  "All information is generated using quantitative models, publicly available data, and historical analysis. Any references to potential trades or market opportunities are hypothetical, generalized, and not tailored to any individual's financial situation, investment objectives, or risk tolerance.",
  "Investing in securities involves substantial risk, including the risk of loss. Past performance is not indicative of future results. You are solely responsible for any investment decisions you make, and you should consult with a licensed financial professional before making any investment decisions.",
  "By using this website, you acknowledge and agree that [Company Name], its owners, operators, and affiliates are not responsible for any financial losses or damages arising from the use of this information.",
];

export default function FinancialDisclaimerPage() {
  return (
    <Container maxWidth="md" sx={{ py: 4, display: "grid", gap: 3 }}>
      <Stack spacing={1}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Financial Disclaimer
        </Typography>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
          <Link component={RouterLink} to="/terms" underline="hover">
            Terms of Service
          </Link>
          <Link component={RouterLink} to="/privacy" underline="hover">
            Privacy Policy
          </Link>
        </Stack>
      </Stack>

      <Paper variant="outlined" sx={{ p: { xs: 2.5, md: 3 }, display: "grid", gap: 2 }}>
        {PARAGRAPHS.map((paragraph) => (
          <Typography key={paragraph} variant="body1" sx={{ lineHeight: 1.7 }}>
            {paragraph}
          </Typography>
        ))}
      </Paper>
    </Container>
  );
}
