import React, { createContext, useContext, useEffect, useMemo, useState } from "react";

const SESSION_KEY = "hart_quantitative_research.auth.session";
const API_BASE = process.env.REACT_APP_API_BASE || "";
const TRIAL_DAYS = 30;

function getStorage() {
  if (typeof window === "undefined") {
    return { local: null, session: null };
  }
  return { local: window.localStorage, session: window.sessionStorage };
}

function readJson(storage, key) {
  if (!storage) {
    return null;
  }
  try {
    const raw = storage.getItem(key);
    if (!raw) {
      return null;
    }
    return JSON.parse(raw);
  } catch (err) {
    return null;
  }
}

function writeJson(storage, key, value) {
  if (!storage) {
    return;
  }
  if (value == null) {
    storage.removeItem(key);
    return;
  }
  storage.setItem(key, JSON.stringify(value));
}

function loadSession() {
  const { local, session } = getStorage();
  const sessionUser = readJson(session, SESSION_KEY);
  if (sessionUser) {
    return { user: sessionUser, mode: "session" };
  }
  const localUser = readJson(local, SESSION_KEY);
  if (localUser) {
    return { user: localUser, mode: "local" };
  }
  return { user: null, mode: null };
}

function saveSession(user, mode) {
  const { local, session } = getStorage();
  if (!user || !mode) {
    writeJson(local, SESSION_KEY, null);
    writeJson(session, SESSION_KEY, null);
    return;
  }
  if (mode === "local") {
    writeJson(local, SESSION_KEY, user);
    writeJson(session, SESSION_KEY, null);
    return;
  }
  writeJson(session, SESSION_KEY, user);
  writeJson(local, SESSION_KEY, null);
}

function normalizeEmail(email) {
  return (email || "").trim().toLowerCase();
}

function formatNameFromEmail(email) {
  const handle = email.split("@")[0] || "member";
  const parts = handle.split(/[._-]+/).filter(Boolean);
  if (!parts.length) {
    return "Member";
  }
  return parts.map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function applyTrialInfo(user) {
  if (!user || !user.createdAt) {
    return user;
  }
  const tier = user.subscriptionTier || "trial";
  if (tier !== "trial" && tier !== "free") {
    return user;
  }
  const createdAt = new Date(user.createdAt);
  if (Number.isNaN(createdAt.getTime())) {
    return user;
  }
  const trialEndsAt = new Date(createdAt.getTime() + TRIAL_DAYS * 24 * 60 * 60 * 1000);
  const msLeft = trialEndsAt.getTime() - Date.now();
  const trialDaysLeft = Math.max(0, Math.ceil(msLeft / (24 * 60 * 60 * 1000)));
  const subscriptionStatus = msLeft <= 0 ? "expired" : user.subscriptionStatus || "trialing";
  return { ...user, subscriptionTier: "trial", subscriptionStatus, trialEndsAt: trialEndsAt.toISOString(), trialDaysLeft };
}

function normalizeUser(rawUser) {
  if (!rawUser) {
    return null;
  }
  const email = normalizeEmail(rawUser.email);
  if (!email) {
    return null;
  }
  const base = {
    ...rawUser,
    email,
    fullName: rawUser.fullName?.trim() || formatNameFromEmail(email),
    subscriptionTier: rawUser.subscriptionTier || "trial",
    subscriptionStatus: rawUser.subscriptionStatus || "active",
  };
  return applyTrialInfo(base);
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const initialSession = loadSession();
  const [user, setUser] = useState(normalizeUser(initialSession.user));
  const [sessionMode, setSessionMode] = useState(initialSession.mode);

  useEffect(() => {
    saveSession(user, sessionMode);
  }, [user, sessionMode]);

  const login = async (email, password, options = {}) => {
    const normalizedEmail = normalizeEmail(email);
    if (!normalizedEmail || !password) {
      return { ok: false, error: "Email and password are required." };
    }
    try {
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: normalizedEmail, password }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        return { ok: false, error: detail?.detail || "Login failed." };
      }
      const data = await res.json();
      const nextUser = normalizeUser(data);
      if (!nextUser) {
        return { ok: false, error: "Invalid login response." };
      }
      setUser(nextUser);
      setSessionMode(options.remember === false ? "session" : "local");
      return { ok: true, user: nextUser };
    } catch (err) {
      return { ok: false, error: "Unable to reach the server." };
    }
  };

  const register = async (email, password, options = {}) => {
    const normalizedEmail = normalizeEmail(email);
    if (!normalizedEmail || !password) {
      return { ok: false, error: "Email and password are required." };
    }
    try {
      const res = await fetch(`${API_BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: normalizedEmail,
          password,
          accepted_terms: Boolean(options.acceptedTerms),
          acknowledged_disclaimer: Boolean(options.acknowledgedDisclaimer),
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        return { ok: false, error: detail?.detail || "Registration failed." };
      }
      const data = await res.json();
      const nextUser = normalizeUser(data);
      if (!nextUser) {
        return { ok: false, error: "Invalid registration response." };
      }
      setUser(nextUser);
      setSessionMode(options.remember === false ? "session" : "local");
      return { ok: true, user: nextUser };
    } catch (err) {
      return { ok: false, error: "Unable to reach the server." };
    }
  };

  const updateUser = (patch) => {
    if (!user) {
      return;
    }
    const nextUser = { ...user, ...patch };
    setUser(nextUser);
  };

  const logout = () => {
    setUser(null);
    setSessionMode(null);
  };

  const deleteAccount = () => {
    if (!user) {
      return;
    }
    setUser(null);
    setSessionMode(null);
  };

  const value = useMemo(
    () => ({
      user,
      isLoggedIn: Boolean(user),
      login,
      register,
      updateUser,
      logout,
      deleteAccount,
    }),
    [user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
