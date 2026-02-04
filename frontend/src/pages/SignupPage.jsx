import React, { useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Chip,
  Container,
  FormControlLabel,
  FormHelperText,
  Link,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { CheckCircle } from "@mui/icons-material";
import { alpha, useTheme } from "@mui/material/styles";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

const TRIAL_FEATURES = [
  "Full access to trading dashboards",
  "Daily stock + options ideas",
  "Cancel anytime during the trial",
];

const emptyForm = {
  email: "",
  password: "",
};

export default function SignupPage() {
  const theme = useTheme();
  const navigate = useNavigate();
  const { register } = useAuth();
  const [form, setForm] = useState(emptyForm);
  const [error, setError] = useState("");
  const [agreementError, setAgreementError] = useState("");
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const updateField = (field) => (event) => {
    const value = event.target.value;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setAgreementError("");
    if (!form.email || !form.password) {
      setError("Email and password are required.");
      return;
    }
    if (!agreedToTerms) {
      setAgreementError("You must agree to the Terms of Service and Financial Disclaimer.");
      return;
    }
    setIsSubmitting(true);
    const result = await register(form.email, form.password, {
      acceptedTerms: true,
      acknowledgedDisclaimer: true,
    });
    setIsSubmitting(false);
    if (!result?.ok) {
      setError(result?.error || "Unable to create your account.");
      return;
    }
    navigate("/profile");
  };

  return (
    <Container maxWidth="md" sx={{ py: 4, display: "grid", gap: 3 }}>
      <Box>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Start your 30-day free trial
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Create your account with an email and password. No payment details required today.
        </Typography>
      </Box>

      <Stack direction={{ xs: "column", md: "row" }} spacing={3}>
        <Paper
          variant="outlined"
          sx={{
            flex: 1,
            p: 3,
            background: `linear-gradient(130deg, ${alpha(
              theme.palette.primary.main,
              0.08,
            )}, ${alpha(theme.palette.secondary.main, 0.18)})`,
            borderColor: alpha(theme.palette.primary.main, 0.2),
          }}
        >
          <Stack spacing={2}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                Free Trial
              </Typography>
              <Chip label="30 days" color="secondary" size="small" />
            </Box>
            <Typography variant="body2" color="text.secondary">
              Explore the full Hart Quantitative Research experience before you subscribe.
            </Typography>
            <List dense disablePadding>
              {TRIAL_FEATURES.map((feature) => (
                <ListItem key={feature} disableGutters>
                  <ListItemIcon sx={{ minWidth: 28 }}>
                    <CheckCircle fontSize="small" color="secondary" />
                  </ListItemIcon>
                  <ListItemText primary={feature} primaryTypographyProps={{ variant: "body2" }} />
                </ListItem>
              ))}
            </List>
          </Stack>
        </Paper>

        <Paper
          component="form"
          onSubmit={handleSubmit}
          variant="outlined"
          sx={{ flex: 1, p: 3, display: "grid", gap: 2 }}
        >
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Create your account
          </Typography>
          <TextField
            label="Email"
            type="email"
            value={form.email}
            onChange={updateField("email")}
            autoComplete="email"
            fullWidth
            required
          />
          <TextField
            label="Password"
            type="password"
            value={form.password}
            onChange={updateField("password")}
            autoComplete="new-password"
            fullWidth
            required
          />
          <Box>
            <FormControlLabel
              control={
                <Checkbox
                  checked={agreedToTerms}
                  onChange={(event) => {
                    setAgreedToTerms(event.target.checked);
                    if (event.target.checked) {
                      setAgreementError("");
                    }
                  }}
                  required
                />
              }
              label={
                <Typography variant="body2">
                  I agree to the{" "}
                  <Link component={RouterLink} to="/terms" underline="hover">
                    Terms of Service
                  </Link>{" "}
                  and acknowledge the{" "}
                  <Link component={RouterLink} to="/disclaimer" underline="hover">
                    Financial Disclaimer
                  </Link>
                </Typography>
              }
            />
            {agreementError ? <FormHelperText error>{agreementError}</FormHelperText> : null}
            <FormHelperText>
              Review the{" "}
              <Link component={RouterLink} to="/privacy" underline="hover">
                Privacy Policy
              </Link>
              .
            </FormHelperText>
          </Box>
          {error ? (
            <Typography variant="body2" color="error">
              {error}
            </Typography>
          ) : null}
          <Button variant="contained" type="submit" size="large" disabled={isSubmitting}>
            {isSubmitting ? "Creating..." : "Create account"}
          </Button>
          <Button variant="text" onClick={() => navigate("/login")} sx={{ justifySelf: "flex-start" }}>
            Already have an account? Log in
          </Button>
        </Paper>
      </Stack>
    </Container>
  );
}
