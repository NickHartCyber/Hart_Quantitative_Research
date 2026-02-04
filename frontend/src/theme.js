import { alpha, createTheme } from "@mui/material/styles";

const navy = "#0f2747";
const navyDark = "#08162c";
const navyLift = "#1f3b63";
const accentBlue = "#5fa8e9";
const silver = "#f2f4f7";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: navy,
      light: navyLift,
      dark: navyDark,
      contrastText: "#e8edf5",
    },
    secondary: {
      main: accentBlue,
      light: "#87c2ff",
      dark: "#2e74b9",
      contrastText: "#0a172a",
    },
    background: {
      default: silver,
      paper: "#ffffff",
    },
    text: {
      primary: "#0f172a",
      secondary: "#4b5563",
    },
    divider: alpha(navy, 0.12),
  },
  shape: { borderRadius: 12 },
  typography: {
    fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
    fontWeightMedium: 600,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: silver,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: `linear-gradient(120deg, ${navyDark}, ${navy})`,
          color: "#e8edf5",
          boxShadow: "0 10px 30px rgba(8, 22, 44, 0.35)",
          borderBottom: `1px solid ${alpha("#ffffff", 0.08)}`,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: "linear-gradient(180deg, #0b1a2f, #0d213a)",
          color: "#e8edf5",
          borderRight: "1px solid rgba(255, 255, 255, 0.08)",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 14,
          borderColor: "rgba(12, 31, 58, 0.08)",
          boxShadow: "0 15px 45px rgba(12, 31, 58, 0.06)",
        },
      },
    },
    MuiButton: {
      defaultProps: {
        disableElevation: true,
      },
      styleOverrides: {
        root: {
          borderRadius: 10,
          textTransform: "none",
          fontWeight: 600,
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          "&.Mui-selected": {
            backgroundColor: alpha(accentBlue, 0.18),
            color: "#f6f9fc",
            "&:hover": {
              backgroundColor: alpha(accentBlue, 0.24),
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 600,
        },
      },
    },
  },
});

export default theme;
