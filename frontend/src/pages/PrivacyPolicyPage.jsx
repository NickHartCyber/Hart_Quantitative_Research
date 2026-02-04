import React from "react";
import { Box, Container, Link, Paper, Stack, Typography } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const API_BASE = process.env.REACT_APP_API_BASE || "";
const PRIVACY_PDF_URL = `${API_BASE}/api/legal/privacy`;

export default function PrivacyPolicyPage() {
  return (
    <Container maxWidth="md" sx={{ py: 4, display: "grid", gap: 3 }}>
      <Stack spacing={1}>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Privacy Policy
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Read or download the latest Privacy Policy.
        </Typography>
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
          <Link component={RouterLink} to="/terms" underline="hover">
            Terms of Service
          </Link>
          <Link component={RouterLink} to="/disclaimer" underline="hover">
            Financial Disclaimer
          </Link>
        </Stack>
      </Stack>
      <Paper variant="outlined" sx={{ overflow: "hidden" }}>
        <Box sx={{ width: "100%", height: { xs: "65vh", md: "72vh" } }}>
          <iframe title="Privacy Policy" src={PRIVACY_PDF_URL} style={{ border: 0, width: "100%", height: "100%" }} />
        </Box>
      </Paper>
    </Container>
  );
}
