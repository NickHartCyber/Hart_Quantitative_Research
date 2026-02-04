import React, { useState } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Container,
  IconButton,
  Divider,
  Button,
  Stack,
  Link as MuiLink,
} from "@mui/material";
import {
  Dashboard as DashboardIcon,
  Menu as MenuIcon,
  Home as HomeIcon,
  AccountCircle,
  Login,
  Paid,
  Search,
  CompareArrows,
  MenuBook,
  MenuOpen,
  Assessment,
  ShowChart,
  Gavel,
} from "@mui/icons-material";
import { alpha, useTheme } from "@mui/material/styles";
import {
  BrowserRouter,
  Routes,
  Route,
  useLocation,
  useNavigate,
  Link,
  Navigate,
  Outlet,
} from "react-router-dom";
import DashboardPage from "./pages/DashboardPage.jsx";
import HomePage from "./pages/HomePage.jsx";
import BuySellSuggesterPage from "./pages/BuySellSuggesterPage.jsx";
import TickerAnalysisPage from "./pages/TickerSearchPage.jsx";
import DefinitionsPage from "./pages/DefinitionsPage.jsx";
import ProfilePage from "./pages/ProfilePage.jsx";
import HoldingsAnalysisPage from "./pages/HoldingsAnalysisPage.jsx";
import OptionsSuggesterPage from "./pages/OptionsSuggesterPage.jsx";
import PoliticianTradesPage from "./pages/PoliticianTradesPage.jsx";
import TickerFundamentalsPage from "./pages/TickerFundamentalsPage.jsx";
import SignupPage from "./pages/SignupPage.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import FinancialDisclaimerPage from "./pages/FinancialDisclaimerPage.jsx";
import TermsOfServicePage from "./pages/TermsOfServicePage.jsx";
import PrivacyPolicyPage from "./pages/PrivacyPolicyPage.jsx";
import { useAuth } from "./auth/AuthContext";
import FinancialDisclaimerNotice from "./components/FinancialDisclaimerNotice.jsx";


// ---- Layout constants --------------------------------------------------------
const drawerWidth = 260;

// ---- Sidebar + Layout --------------------------------------------------------
const navItems = [
  { label: "Home", icon: <HomeIcon />, to: "/" },
  { label: "Stock Ideas", icon: <CompareArrows />, to: "/suggester" },
  { label: "Options Ideas", icon: <ShowChart />, to: "/options" },
  { label: "Holdings Checkup", icon: <Assessment />, to: "/holdings" },
  { label: "Ticker Analysis", icon: <Search />, to: "/ticker" },
  { label: "Politician Trades", icon: <Gavel />, to: "/politician-trades" },
  { label: "Nick's Trades", icon: <DashboardIcon />, to: "/dashboard" },
  { label: "Definitions", icon: <MenuBook />, to: "/definitions" },
];

const DISCLAIMER_ROUTES = [
  "/",
  "/suggester",
  "/options",
  "/ticker",
  "/holdings",
  "/dashboard",
  "/politician-trades",
];

const shouldShowDisclaimer = (pathname) =>
  DISCLAIMER_ROUTES.some((route) => (route === "/" ? pathname === "/" : pathname.startsWith(route)));

function Sidebar({ mobileOpen, onClose, desktopOpen }) {
  const location = useLocation();
  const navigate = useNavigate();
  const theme = useTheme();
  const navActiveColor = "#f6f9fc";
  const navMutedColor = "rgba(232, 237, 245, 0.7)";
  const selectedBg = alpha(theme.palette.secondary.main, 0.22);
  const hoverBg = alpha("#ffffff", 0.08);

  const list = (
    <Box role="presentation" sx={{ width: drawerWidth }}>
      <Toolbar sx={{ px: 3, pt: 2, pb: 1 }}>
        <Typography variant="h6" noWrap sx={{ fontWeight: 700, letterSpacing: 0.4 }}>
          Hart Quantitative Research
        </Typography>
      </Toolbar>
      <Divider sx={{ borderColor: "rgba(255,255,255,0.08)" }} />
      <List sx={{ px: 1, pt: 1 }}>
        {navItems.map((item) => {
          const selected =
            item.to === "/"
              ? location.pathname === "/"
              : location.pathname.startsWith(item.to);
          return (
            <ListItem key={item.to} disablePadding>
              <ListItemButton
                selected={selected}
                onClick={() => navigate(item.to)}
                sx={{
                  mx: 0.5,
                  my: 0.35,
                  borderRadius: 1.5,
                  color: navMutedColor,
                  transition: "background-color 120ms ease, color 120ms ease",
                  "& .MuiListItemIcon-root": { color: "inherit" },
                  "& .MuiListItemText-primary": { fontWeight: 600, letterSpacing: 0.2 },
                  "&.Mui-selected": {
                    color: navActiveColor,
                    backgroundColor: selectedBg,
                  },
                  "&.Mui-selected:hover": {
                    backgroundColor: alpha(theme.palette.secondary.main, 0.3),
                  },
                  "&:hover": {
                    backgroundColor: hoverBg,
                    color: navActiveColor,
                  },
                }}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Box>
  );

  return (
    <Box
      component="nav"
      sx={{ width: { sm: desktopOpen ? drawerWidth : 0 }, flexShrink: { sm: 0 } }}
      aria-label="sidebar"
    >
      {/* Mobile drawer */}
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onClose}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: "block", sm: "none" },
          "& .MuiDrawer-paper": { boxSizing: "border-box", width: drawerWidth, borderRight: "none" },
        }}
      >
        {list}
      </Drawer>
      {/* Desktop drawer */}
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: "none", sm: desktopOpen ? "block" : "none" },
          "& .MuiDrawer-paper": { boxSizing: "border-box", width: drawerWidth, borderRight: "none" },
        }}
        open
      >
        {list}
      </Drawer>
    </Box>
  );
}

function Layout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [desktopOpen, setDesktopOpen] = useState(true);
  const { isLoggedIn } = useAuth();
  const location = useLocation();
  const handleMobileToggle = () => setMobileOpen((x) => !x);
  const handleDesktopToggle = () => setDesktopOpen((x) => !x);
  const sidebarOffset = desktopOpen ? drawerWidth : 0;
  const showDisclaimer = shouldShowDisclaimer(location.pathname);

  return (
    <Box sx={{ display: "flex" }}>
      <AppBar
        position="fixed"
        color="primary"
        sx={{
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backdropFilter: "blur(8px)",
        }}
      >
        <Toolbar sx={{ gap: 2 }}>
          <IconButton color="inherit" edge="start" onClick={handleMobileToggle} sx={{ display: { sm: "none" } }}>
            <MenuIcon />
          </IconButton>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDesktopToggle}
            sx={{ display: { xs: "none", sm: "inline-flex" } }}
            aria-label={desktopOpen ? "Hide sidebar" : "Show sidebar"}
          >
            {desktopOpen ? <MenuOpen /> : <MenuIcon />}
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 700 }}>
            Hart Quantitative Research
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          {!isLoggedIn && (
            <>
              <Button
                color="inherit"
                startIcon={<Paid />}
                component={Link}
                to="/signup"
                sx={{ textTransform: "none", border: "1px solid rgba(255,255,255,0.3)", mr: 1 }}
              >
                Subscribe
              </Button>
              <Button
                color="inherit"
                startIcon={<Login />}
                component={Link}
                to="/login"
                sx={{ textTransform: "none", mr: 1 }}
              >
                Log in
              </Button>
            </>
          )}
          {isLoggedIn && (
            <Button
              color="inherit"
              startIcon={<AccountCircle />}
              component={Link}
              to="/profile"
              sx={{ textTransform: "none" }}
            >
              Profile
            </Button>
          )}
        </Toolbar>
      </AppBar>

      <Sidebar mobileOpen={mobileOpen} onClose={handleMobileToggle} desktopOpen={desktopOpen} />

      <Box
        component="main"
        sx={(theme) => ({
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${sidebarOffset}px)` },
          minHeight: "100vh",
          backgroundColor: theme.palette.background.default,
          backgroundImage: `radial-gradient(circle at 20% 20%, ${alpha(
            theme.palette.primary.main,
            0.05
          )}, transparent 30%), radial-gradient(circle at 80% 0%, ${alpha(
            theme.palette.secondary.main,
            0.08
          )}, transparent 25%)`,
        })}
      >
        <Toolbar />
        {children}
        {showDisclaimer && (
          <Container maxWidth="lg" sx={{ mt: 4 }}>
            <FinancialDisclaimerNotice />
          </Container>
        )}
        <Divider sx={{ my: 4, borderColor: "rgba(0,0,0,0.08)" }} />
        <Container maxWidth="lg" sx={{ pb: 4 }}>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ xs: "flex-start", sm: "center" }}>
            <Typography variant="caption" color="text.secondary">
              Legal
            </Typography>
            <MuiLink component={Link} to="/terms" underline="hover">
              Terms of Service
            </MuiLink>
            <MuiLink component={Link} to="/privacy" underline="hover">
              Privacy Policy
            </MuiLink>
            <MuiLink component={Link} to="/disclaimer" underline="hover">
              Financial Disclaimer
            </MuiLink>
          </Stack>
        </Container>
      </Box>
    </Box>
  );
}

function RequireAuth() {
  const { user } = useAuth();
  const location = useLocation();

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  if (user.subscriptionStatus === "expired" && location.pathname !== "/profile") {
    return <Navigate to="/profile" replace />;
  }

  return <Outlet />;
}

// ---- App entry with routes ---------------------------------------------------
export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/definitions" element={<DefinitionsPage />} />
          <Route element={<RequireAuth />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/ticker" element={<TickerAnalysisPage />} />
            <Route path="/ticker/fundamentals" element={<TickerFundamentalsPage />} />
            <Route path="/suggester" element={<BuySellSuggesterPage />} />
            <Route path="/options" element={<OptionsSuggesterPage />} />
            <Route path="/holdings" element={<HoldingsAnalysisPage />} />
            <Route path="/politician-trades" element={<PoliticianTradesPage />} />
            <Route path="/profile" element={<ProfilePage />} />
          </Route>
          <Route path="/signup" element={<SignupPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/disclaimer" element={<FinancialDisclaimerPage />} />
          <Route path="/terms" element={<TermsOfServicePage />} />
          <Route path="/privacy" element={<PrivacyPolicyPage />} />
          {/* Add more routes as needed */}
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
