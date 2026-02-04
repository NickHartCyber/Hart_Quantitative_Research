import React, { useState } from "react";
import {
  Box,
  Button,
  Checkbox,
  Container,
  FormControlLabel,
  Link,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useLocation, useNavigate } from "react-router-dom";
import { alpha, useTheme } from "@mui/material/styles";
import { useAuth } from "../auth/AuthContext";

const emptyForm = { email: "", password: "", remember: true };

export default function LoginPage() {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const { user, login } = useAuth();
  const [form, setForm] = useState(emptyForm);
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const updateField = (field) => (event) => {
    const value = event.target.type === "checkbox" ? event.target.checked : event.target.value;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    const email = form.email.trim();
    if (!email || !form.password) {
      setError("Email and password are required.");
      return;
    }
    setIsSubmitting(true);
    const result = await login(email, form.password, { remember: form.remember });
    setIsSubmitting(false);
    if (!result?.ok) {
      setError(result?.error || "Unable to log in.");
      return;
    }
    const fallback = { pathname: "/profile" };
    const destination = location.state?.from?.pathname ? location.state.from : fallback;
    navigate(destination, { replace: true });
  };

  if (user) {
    return (
      <Container maxWidth="sm" sx={{ py: 6, display: "grid", gap: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            You're already signed in
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Signed in as {user.email}.
          </Typography>
        </Box>
        <Paper variant="outlined" sx={{ p: 3, display: "grid", gap: 2 }}>
          <Button variant="contained" size="large" onClick={() => navigate("/profile")}>
            Go to profile
          </Button>
        </Paper>
      </Container>
    );
  }

  return (
    <Container maxWidth="sm" sx={{ py: 6, display: "grid", gap: 3 }}>
      <Box>
        <Typography variant="h4" sx={{ fontWeight: 700 }}>
          Welcome back
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Log in to manage your subscription and trading tools.
        </Typography>
      </Box>

      <Paper
        component="form"
        onSubmit={handleSubmit}
        variant="outlined"
        sx={{
          p: 3,
          display: "grid",
          gap: 2,
          background: `linear-gradient(140deg, ${alpha(theme.palette.primary.main, 0.06)}, #ffffff)`,
        }}
      >
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
          autoComplete="current-password"
          fullWidth
          required
        />
        {error ? (
          <Typography variant="body2" color="error">
            {error}
          </Typography>
        ) : null}
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <FormControlLabel
            control={<Checkbox checked={form.remember} onChange={updateField("remember")} />}
            label="Remember me"
          />
          <Link
            component="button"
            type="button"
            onClick={() => navigate("/signup")}
            underline="hover"
            sx={{ fontSize: 14 }}
          >
            Need an account?
          </Link>
        </Stack>
        <Button variant="contained" type="submit" size="large" disabled={isSubmitting}>
          {isSubmitting ? "Logging in..." : "Log in"}
        </Button>
      </Paper>
    </Container>
  );
}
