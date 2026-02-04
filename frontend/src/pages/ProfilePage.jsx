import React from "react";
import {
  Box,
  Button,
  Chip,
  Container,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { AccountCircle, DeleteForever, Logout, ManageAccounts } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { alpha, useTheme } from "@mui/material/styles";

const PLAN_LABELS = {
  trial: "Free trial",
  pro: "Pro",
  elite: "Elite",
};

function InfoItem({ label, value }) {
  return (
    <Box sx={{ display: "grid", gap: 0.4 }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
        {value || "—"}
      </Typography>
    </Box>
  );
}

export default function ProfilePage() {
  const theme = useTheme();
  const navigate = useNavigate();
  const { user, logout, deleteAccount } = useAuth();

  if (!user) {
    return (
      <Container maxWidth="sm" sx={{ py: 6, display: "grid", gap: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            You're logged out
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Log in to view your profile and membership details.
          </Typography>
        </Box>
        <Paper variant="outlined" sx={{ p: 3, display: "grid", gap: 2 }}>
          <Button variant="contained" size="large" onClick={() => navigate("/login")}>
            Go to login
          </Button>
        </Paper>
      </Container>
    );
  }

  const tier = user.subscriptionTier === "free" ? "trial" : user.subscriptionTier;
  const planLabel = PLAN_LABELS[tier] || tier || "Plan";
  const isPro = tier === "pro";
  const isTrial = tier === "trial";
  const memberSince = user.createdAt ? new Date(user.createdAt).toLocaleDateString() : "—";
  const trialEnds = user.trialEndsAt ? new Date(user.trialEndsAt).toLocaleDateString() : "—";
  const trialDaysLeft = typeof user.trialDaysLeft === "number" ? `${user.trialDaysLeft} days` : "—";
  const isExpired = user.subscriptionStatus === "expired";

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const handleDelete = () => {
    const confirmed = window.confirm("Delete your account? This will sign you out and remove saved profile data.");
    if (!confirmed) {
      return;
    }
    deleteAccount();
    navigate("/");
  };

  return (
    <Container maxWidth="md" sx={{ py: 4, display: "grid", gap: 3 }}>
      <Box>
        <Typography variant="h5" sx={{ fontWeight: 700 }}>
          Profile
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Your account details and membership status.
        </Typography>
      </Box>

      <Paper
        variant="outlined"
        sx={{
          p: 3,
          display: "grid",
          gap: 2,
          background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.08)}, #ffffff)`,
        }}
      >
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ sm: "center" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
            <AccountCircle color="primary" />
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 700 }}>
                {user.fullName || "Member"}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {user.email}
              </Typography>
            </Box>
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          <Chip
            label={isExpired ? "Trial expired" : `${planLabel} plan`}
            color={isPro ? "secondary" : isExpired ? "error" : "default"}
            sx={{ fontWeight: 600 }}
          />
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 3, display: "grid", gap: 2 }}>
        <Typography variant="h6">User information</Typography>
        <Box
          sx={{
            display: "grid",
            gap: 2,
            gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))" },
          }}
        >
          <InfoItem label="Name" value={user.fullName} />
          <InfoItem label="Email" value={user.email} />
          <InfoItem label="Subscription tier" value={planLabel} />
          <InfoItem label="Status" value={user.subscriptionStatus || "active"} />
          {isTrial && <InfoItem label="Trial ends" value={trialEnds} />}
          {isTrial && <InfoItem label="Trial days left" value={trialDaysLeft} />}
          <InfoItem label="Member since" value={memberSince} />
        </Box>
      </Paper>

      <Paper variant="outlined" sx={{ p: 3, display: "grid", gap: 2 }}>
        <Box sx={{ display: "grid", gap: 0.5 }}>
          <Typography variant="h6">Membership</Typography>
          <Typography variant="body2" color="text.secondary">
            Manage your subscription and account access.
          </Typography>
        </Box>
        {isExpired && (
          <Typography variant="body2" color="error">
            Your free trial has ended. Please upgrade to regain access.
          </Typography>
        )}
        <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5} alignItems="flex-start">
          {isPro && (
            <Button variant="contained" startIcon={<ManageAccounts />} onClick={() => navigate("/signup")}>
              Upgrade membership
            </Button>
          )}
          {isPro && (
            <Button variant="outlined" color="error" startIcon={<DeleteForever />} onClick={handleDelete}>
              Delete account
            </Button>
          )}
          <Button variant="text" startIcon={<Logout />} onClick={handleLogout}>
            Log out
          </Button>
        </Stack>
      </Paper>
    </Container>
  );
}
