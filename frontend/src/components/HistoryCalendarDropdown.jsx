import React, { useMemo, useState } from "react";
import {
  Box,
  IconButton,
  InputAdornment,
  Popover,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { ChevronLeft, ChevronRight, Event } from "@mui/icons-material";

const WEEK_DAYS = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

function normalizeDateKey(value) {
  if (typeof value !== "string") {
    return "";
  }
  return value.replace(/-/g, "").trim();
}

function formatHistoryDateLabel(dateStr) {
  if (typeof dateStr !== "string") {
    return "";
  }
  if (dateStr.length === 8) {
    return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6)}`;
  }
  return dateStr;
}

function formatDateKey(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}${month}${day}`;
}

function parseDateKey(dateKey) {
  if (!dateKey || dateKey.length !== 8) {
    return null;
  }
  const year = Number(dateKey.slice(0, 4));
  const month = Number(dateKey.slice(4, 6));
  const day = Number(dateKey.slice(6, 8));
  if (!year || !month || !day) {
    return null;
  }
  const date = new Date(year, month - 1, day);
  if (Number.isNaN(date.valueOf())) {
    return null;
  }
  return date;
}

function getMonthLabel(date) {
  return date.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function getMonthCells(date) {
  const year = date.getFullYear();
  const month = date.getMonth();
  const firstOfMonth = new Date(year, month, 1);
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const offset = firstOfMonth.getDay();
  const cells = [];
  for (let i = 0; i < offset; i += 1) {
    cells.push(null);
  }
  for (let day = 1; day <= daysInMonth; day += 1) {
    cells.push(day);
  }
  const remainder = cells.length % 7;
  if (remainder) {
    for (let i = 0; i < 7 - remainder; i += 1) {
      cells.push(null);
    }
  }
  return { year, month, cells };
}

export default function HistoryCalendarDropdown({
  label,
  value,
  onChange,
  historyRuns,
  helperText,
  disabled = false,
  latestLabel = "Latest (cached)",
  latestValue = "latest",
  minWidth = 220,
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [displayMonth, setDisplayMonth] = useState(() => new Date());

  const { availableDates, latestKey } = useMemo(() => {
    const set = new Set();
    (historyRuns || []).forEach((entry) => {
      const key = normalizeDateKey(entry?.date || entry?.label);
      if (key.length === 8) {
        set.add(key);
      }
    });
    const keys = Array.from(set).sort();
    return { availableDates: set, latestKey: keys[keys.length - 1] || "" };
  }, [historyRuns]);

  const selectedKey =
    value === latestValue ? latestKey : normalizeDateKey(value);
  const selectedLabel =
    value === latestValue ? latestLabel : formatHistoryDateLabel(value);
  const open = Boolean(anchorEl);
  const monthLabel = getMonthLabel(displayMonth);
  const { year, month, cells } = useMemo(() => getMonthCells(displayMonth), [displayMonth]);

  const handleOpen = (event) => {
    const baseKey = selectedKey || latestKey;
    const baseDate = parseDateKey(baseKey) || new Date();
    setDisplayMonth(new Date(baseDate.getFullYear(), baseDate.getMonth(), 1));
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleSelect = (dateKey) => {
    if (!dateKey || !availableDates.has(dateKey)) {
      return;
    }
    onChange(dateKey);
    setAnchorEl(null);
  };

  const handlePrevMonth = () => {
    setDisplayMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() - 1, 1));
  };

  const handleNextMonth = () => {
    setDisplayMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() + 1, 1));
  };

  return (
    <>
      <TextField
        size="small"
        label={label}
        value={selectedLabel}
        onClick={handleOpen}
        helperText={helperText}
        disabled={disabled}
        InputProps={{
          readOnly: true,
          endAdornment: (
            <InputAdornment position="end">
              <Event fontSize="small" color={disabled ? "disabled" : "action"} />
            </InputAdornment>
          ),
        }}
        sx={{ minWidth }}
      />
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Box sx={{ p: 2, width: 280 }}>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
            <IconButton size="small" onClick={handlePrevMonth} aria-label="Previous month">
              <ChevronLeft fontSize="small" />
            </IconButton>
            <Typography variant="subtitle2">{monthLabel}</Typography>
            <IconButton size="small" onClick={handleNextMonth} aria-label="Next month">
              <ChevronRight fontSize="small" />
            </IconButton>
          </Stack>
          <Box sx={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 0.5 }}>
            {WEEK_DAYS.map((day) => (
              <Typography key={day} variant="caption" color="text.secondary" sx={{ textAlign: "center" }}>
                {day}
              </Typography>
            ))}
            {cells.map((day, idx) => {
              if (!day) {
                return <Box key={`empty-${idx}`} sx={{ height: 32 }} />;
              }
              const dateKey = formatDateKey(new Date(year, month, day));
              const isAvailable = availableDates.has(dateKey);
              const isSelected = selectedKey === dateKey;
              return (
                <Box
                  key={dateKey}
                  component="button"
                  type="button"
                  onClick={() => handleSelect(dateKey)}
                  disabled={!isAvailable}
                  sx={{
                    border: "1px solid",
                    borderColor: isSelected ? "primary.main" : "divider",
                    borderRadius: 1,
                    bgcolor: isSelected ? "primary.main" : isAvailable ? "action.selected" : "transparent",
                    color: isSelected ? "primary.contrastText" : "text.primary",
                    width: 36,
                    height: 32,
                    fontSize: 13,
                    cursor: isAvailable ? "pointer" : "default",
                    opacity: isAvailable ? 1 : 0.35,
                    "&:hover": isAvailable
                      ? { bgcolor: isSelected ? "primary.dark" : "action.hover" }
                      : undefined,
                  }}
                >
                  {day}
                </Box>
              );
            })}
          </Box>
          {!availableDates.size && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
              No saved runs yet.
            </Typography>
          )}
        </Box>
      </Popover>
    </>
  );
}
