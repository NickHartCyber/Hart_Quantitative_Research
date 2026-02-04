import React, { useEffect, useMemo, useState } from "react";
import { Link as RouterLink, useLocation, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Container,
  Divider,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { ArrowBack, Search } from "@mui/icons-material";

const API_BASE = process.env.REACT_APP_API_BASE || "";

const FUNDAMENTAL_FIELDS = [
  { key: "price", label: "Last Price", type: "currency" },
  { key: "market_cap", label: "Market Cap", type: "compactCurrency" },
  { key: "sector", label: "Sector", type: "text" },
  { key: "pe_ratio", label: "P/E (TTM)", type: "number" },
  { key: "forward_pe", label: "Forward P/E", type: "number" },
  { key: "peg_ratio", label: "PEG Ratio", type: "number" },
  { key: "eps_ttm", label: "EPS (TTM)", type: "number" },
  { key: "roe", label: "Return on Equity", type: "percent" },
  { key: "debt_to_equity", label: "Debt to Equity", type: "number" },
  { key: "dividend_yield", label: "Dividend Yield", type: "percent" },
  { key: "profit_margin", label: "Profit Margin", type: "percent" },
  { key: "book_value", label: "Book Value / Share", type: "currency" },
  { key: "fifty_two_week_high", label: "52w High", type: "currency" },
  { key: "fifty_two_week_low", label: "52w Low", type: "currency" },
  { key: "beta", label: "Beta", type: "number" },
  { key: "shares_outstanding", label: "Shares Outstanding", type: "compactNumber" },
];

const TRADING_FIELDS = [
  { key: "lastPrice", label: "Last Price", type: "currency" },
  { key: "previousClose", label: "Prev Close", type: "currency" },
  { key: "open", label: "Open", type: "currency" },
  { key: "dayLow", label: "Day Low", type: "currency" },
  { key: "dayHigh", label: "Day High", type: "currency" },
  { key: "yearLow", label: "52w Low", type: "currency" },
  { key: "yearHigh", label: "52w High", type: "currency" },
  { key: "lastVolume", label: "Volume", type: "compactNumber" },
  { key: "tenDayAverageVolume", label: "10D Avg Volume", type: "compactNumber" },
  { key: "threeMonthAverageVolume", label: "3M Avg Volume", type: "compactNumber" },
  { key: "fiftyDayAverage", label: "50D Avg", type: "currency" },
  { key: "twoHundredDayAverage", label: "200D Avg", type: "currency" },
  { key: "marketCap", label: "Market Cap", type: "compactCurrency" },
  { key: "shares", label: "Shares Outstanding", type: "compactNumber" },
  { key: "exchange", label: "Exchange", type: "text" },
  { key: "yearChange", label: "YTD Change", type: "percent" },
];

const METRIC_INFO = {
  price: {
    definition: "Latest trading price of the stock.",
    calc: "Most recent trade price reported by the exchange.",
    range: "No fixed range; compare to 52w range and trend.",
  },
  market_cap: {
    definition: "Total market value of all shares (price x shares).",
    calc: "Share price x shares outstanding.",
    range: "Higher is typically more stable; compare to peers.",
  },
  sector: {
    definition: "Industry group the company operates in.",
    calc: "Categorization assigned by market data providers.",
    range: "N/A (categorical).",
  },
  pe_ratio: {
    definition: "Price divided by trailing 12-month earnings per share.",
    calc: "Share price / EPS (TTM).",
    range: "Often 10-25 for mature firms; higher for growth.",
  },
  forward_pe: {
    definition: "Price divided by expected next-year earnings per share.",
    calc: "Share price / forecast EPS.",
    range: "Often 10-25 for mature firms; lower is cheaper.",
  },
  peg_ratio: {
    definition: "P/E divided by earnings growth rate.",
    calc: "P/E / expected EPS growth rate.",
    range: "Around 0.5-1.5 is often reasonable.",
  },
  eps_ttm: {
    definition: "Earnings per share over the last 12 months.",
    calc: "Net income (TTM) / weighted average shares.",
    range: "Healthy if positive and growing year over year.",
  },
  roe: {
    definition: "Return on equity, a measure of profit per equity dollar.",
    calc: "Net income / average shareholder equity.",
    range: "10-20% is solid; >20% is strong (varies by sector).",
  },
  debt_to_equity: {
    definition: "Total debt relative to shareholder equity.",
    calc: "Total debt / shareholder equity.",
    range: "Below 1.0 is commonly healthy; <0.5 conservative.",
  },
  dividend_yield: {
    definition: "Annual dividends divided by current price.",
    calc: "Annual dividend per share / share price.",
    range: "Around 2-5% is often sustainable; very high can be risky.",
  },
  profit_margin: {
    definition: "Net income as a percentage of revenue.",
    calc: "Net income / total revenue.",
    range: "Above 10% is strong; >20% is excellent (sector dependent).",
  },
  book_value: {
    definition: "Shareholder equity per share.",
    calc: "Total equity / shares outstanding.",
    range: "Positive and rising is generally healthier.",
  },
  fifty_two_week_high: {
    definition: "Highest price in the last 52 weeks.",
    calc: "Max daily close over the last 52 weeks.",
    range: "Compare current price to this range.",
  },
  fifty_two_week_low: {
    definition: "Lowest price in the last 52 weeks.",
    calc: "Min daily close over the last 52 weeks.",
    range: "Compare current price to this range.",
  },
  beta: {
    definition: "Volatility vs the market (S&P 500).",
    calc: "Covariance(stock, market) / variance(market).",
    range: "0.8-1.2 is market-like; >1 is more volatile.",
  },
  shares_outstanding: {
    definition: "Total shares issued by the company.",
    calc: "Company-reported share count.",
    range: "Stable or declining is typically healthier.",
  },
  lastPrice: {
    definition: "Latest trading price of the stock.",
    calc: "Most recent trade price reported by the exchange.",
    range: "No fixed range; compare to 52w range and trend.",
  },
  previousClose: {
    definition: "Prior session closing price.",
    calc: "Official close from the previous trading day.",
    range: "No fixed range; compare to current price.",
  },
  open: {
    definition: "Price at the start of the trading session.",
    calc: "First traded price after the open.",
    range: "No fixed range; compare to prior close.",
  },
  dayLow: {
    definition: "Lowest price traded today.",
    calc: "Min trade price during the session.",
    range: "No fixed range; compare to recent range.",
  },
  dayHigh: {
    definition: "Highest price traded today.",
    calc: "Max trade price during the session.",
    range: "No fixed range; compare to recent range.",
  },
  yearLow: {
    definition: "Lowest price in the last 52 weeks.",
    calc: "Min daily close over the last 52 weeks.",
    range: "No fixed range; compare current price to this level.",
  },
  yearHigh: {
    definition: "Highest price in the last 52 weeks.",
    calc: "Max daily close over the last 52 weeks.",
    range: "No fixed range; compare current price to this level.",
  },
  lastVolume: {
    definition: "Most recent trading volume.",
    calc: "Total shares traded in the latest session.",
    range: "Healthy if liquidity is steady and not erratic.",
  },
  tenDayAverageVolume: {
    definition: "Average daily volume over the last 10 trading days.",
    calc: "10-day mean of daily volume.",
    range: "Higher is more liquid; compare to recent volume.",
  },
  threeMonthAverageVolume: {
    definition: "Average daily volume over the last 3 months.",
    calc: "3-month mean of daily volume.",
    range: "Higher is more liquid; compare to recent volume.",
  },
  fiftyDayAverage: {
    definition: "Average price over the last 50 trading days.",
    calc: "50-day mean of daily close.",
    range: "Price above this can signal strength.",
  },
  twoHundredDayAverage: {
    definition: "Average price over the last 200 trading days.",
    calc: "200-day mean of daily close.",
    range: "Price above this can signal long-term strength.",
  },
  marketCap: {
    definition: "Total market value of all shares (price x shares).",
    calc: "Share price x shares outstanding.",
    range: "Higher is typically more stable; compare to peers.",
  },
  shares: {
    definition: "Total shares issued by the company.",
    calc: "Company-reported share count.",
    range: "Stable or declining is typically healthier.",
  },
  exchange: {
    definition: "Primary exchange where the stock trades.",
    calc: "Reported by market data provider.",
    range: "N/A (categorical).",
  },
  yearChange: {
    definition: "Price change over the last year.",
    calc: "Current price / price one year ago - 1.",
    range: "Positive is generally healthy; compare to index.",
  },
};

const STATEMENT_ITEM_HINTS = [
  {
    match: /tax effect of unusual items/i,
    definition: "After-tax impact of one-time or unusual items.",
    calc: "Unusual pre-tax items x effective tax rate.",
    range: "Smaller magnitude is typically healthier.",
  },
  {
    match: /tax rate for calcs|effective tax rate/i,
    definition: "Effective tax rate used for normalizing earnings.",
    calc: "Income tax expense / pre-tax income.",
    range: "Stable and aligned with statutory rates is healthier.",
  },
  {
    match: /normalized ebitda/i,
    definition: "EBITDA adjusted to remove unusual items.",
    calc: "EBITDA +/- non-recurring adjustments.",
    range: "Consistent or rising is healthier.",
  },
  {
    match: /ebitda/i,
    definition: "Earnings before interest, taxes, depreciation, and amortization.",
    calc: "Operating income + D&A (or Net income + interest + taxes + D&A).",
    range: "Positive and growing is healthier.",
  },
  {
    match: /\bebit\b/i,
    definition: "Earnings before interest and taxes.",
    calc: "Net income + interest + taxes.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /net income from continuing operation/i,
    definition: "Profit from ongoing operations attributable to shareholders.",
    calc: "Pre-tax income - taxes - minority interest adjustments.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /net income from continuing and discontinued/i,
    definition: "Total profit including discontinued operations.",
    calc: "Continuing ops net income + discontinued ops net income.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /net income/i,
    definition: "Profit after all expenses, interest, and taxes.",
    calc: "Revenue - all expenses - interest - taxes.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /normalized income/i,
    definition: "Net income adjusted for non-recurring items.",
    calc: "Net income +/- unusual items.",
    range: "Consistent or rising is healthier.",
  },
  {
    match: /interest expense/i,
    definition: "Cost of borrowing for the period.",
    calc: "Interest paid on outstanding debt.",
    range: "Lower relative to operating income is healthier.",
  },
  {
    match: /interest income/i,
    definition: "Income earned on cash and investments.",
    calc: "Interest earned on cash and securities.",
    range: "Stable or rising is healthier.",
  },
  {
    match: /net interest income/i,
    definition: "Interest income minus interest expense.",
    calc: "Interest income - interest expense.",
    range: "Positive is healthier.",
  },
  {
    match: /income before tax|pretax income/i,
    definition: "Profit before income taxes.",
    calc: "Operating income + other income - interest expense.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /income tax expense|taxes/i,
    definition: "Income taxes owed for the period.",
    calc: "Pre-tax income x effective tax rate.",
    range: "Stable relative to pre-tax income is healthier.",
  },
  {
    match: /diluted average shares/i,
    definition: "Weighted average shares including dilution.",
    calc: "Average shares outstanding + dilutive securities.",
    range: "Stable or declining is healthier.",
  },
  {
    match: /basic average shares/i,
    definition: "Weighted average shares outstanding.",
    calc: "Average common shares outstanding during the period.",
    range: "Stable or declining is healthier.",
  },
  {
    match: /diluted eps/i,
    definition: "Earnings per share using diluted share count.",
    calc: "Net income available to common / diluted shares.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /basic eps/i,
    definition: "Earnings per share using basic share count.",
    calc: "Net income available to common / basic shares.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /total revenue|net revenue|total net revenue/i,
    definition: "Total sales for the period.",
    calc: "Sum of all revenue streams.",
    range: "Consistent growth is healthy.",
  },
  {
    match: /reconciled cost of revenue|cost of revenue|cost of goods sold/i,
    definition: "Direct costs to produce goods or services.",
    calc: "Materials + labor + direct costs tied to revenue.",
    range: "Lower relative to revenue is healthier.",
  },
  {
    match: /gross profit/i,
    definition: "Revenue minus cost of revenue.",
    calc: "Total revenue - cost of revenue.",
    range: "Stable or rising gross margin is healthy.",
  },
  {
    match: /operating income|operating profit|total operating income/i,
    definition: "Profit from core operations.",
    calc: "Gross profit - operating expenses.",
    range: "Positive and growing is healthy.",
  },
  {
    match: /operating expenses|total operating expenses/i,
    definition: "Costs to run the business excluding COGS.",
    calc: "SG&A + R&D + other operating costs.",
    range: "Lower relative to revenue is healthier.",
  },
  {
    match: /selling general and administration|sg&a/i,
    definition: "Operating costs outside of production.",
    calc: "Sales + marketing + administrative expenses.",
    range: "Controlled growth vs revenue is healthier.",
  },
  {
    match: /research and development|r&d/i,
    definition: "Spending on product and technology development.",
    calc: "R&D expense recognized for the period.",
    range: "Steady investment is healthy for innovation-heavy firms.",
  },
  {
    match: /reconciled depreciation|depreciation and amortization|depreciation|amortization/i,
    definition: "Non-cash expense allocating asset costs over time.",
    calc: "Periodic allocation of PP&E and intangibles.",
    range: "Stable relative to asset base is healthier.",
  },
  {
    match: /other comprehensive income|gains losses not affecting retained earnings|other equity adjustments/i,
    definition: "Equity changes from items not in net income.",
    calc: "Unrealized gains/losses and other OCI adjustments.",
    range: "Smaller volatility is healthier.",
  },
  {
    match: /total expenses/i,
    definition: "All expenses recognized during the period.",
    calc: "Cost of revenue + operating expenses + other expenses.",
    range: "Lower relative to revenue is healthier.",
  },
  {
    match: /cash and cash equivalents|cash equivalents|cash$/i,
    definition: "Liquid cash available on the balance sheet.",
    calc: "Cash on hand + demand deposits + cash equivalents.",
    range: "Higher relative to short-term obligations is healthier.",
  },
  {
    match: /short term investments|marketable securities/i,
    definition: "Highly liquid investments held short term.",
    calc: "Securities expected to be liquid within a year.",
    range: "Higher adds flexibility; compare to liabilities.",
  },
  {
    match: /accounts receivable|net receivables/i,
    definition: "Amounts owed by customers.",
    calc: "Gross receivables - allowance for bad debt.",
    range: "Lower relative to sales is healthier.",
  },
  {
    match: /inventory/i,
    definition: "Goods held for sale or production.",
    calc: "Cost of inventory on hand.",
    range: "Stable vs sales is healthier; excess can be risky.",
  },
  {
    match: /prepaid|other current assets/i,
    definition: "Short-term assets not in cash or receivables.",
    calc: "Prepaid expenses + other current items.",
    range: "Stable relative to revenue is healthier.",
  },
  {
    match: /current assets/i,
    definition: "Assets expected to convert to cash within a year.",
    calc: "Cash + receivables + inventory + other current assets.",
    range: "Higher vs current liabilities is healthier.",
  },
  {
    match: /property plant and equipment|ppe|net ppe/i,
    definition: "Long-lived tangible assets used in operations.",
    calc: "Gross PP&E - accumulated depreciation.",
    range: "Efficient vs revenue is healthier.",
  },
  {
    match: /goodwill/i,
    definition: "Premium paid in acquisitions above fair value.",
    calc: "Purchase price - fair value of net assets acquired.",
    range: "Lower relative to assets is usually healthier.",
  },
  {
    match: /intangible assets/i,
    definition: "Non-physical assets like patents or trademarks.",
    calc: "Acquisition intangibles + capitalized development costs.",
    range: "Stable relative to revenue is healthier.",
  },
  {
    match: /total assets/i,
    definition: "Everything the company owns.",
    calc: "Current assets + non-current assets.",
    range: "Stable growth is healthy; compare to revenue.",
  },
  {
    match: /accounts payable|payables/i,
    definition: "Amounts owed to suppliers.",
    calc: "Unpaid supplier invoices.",
    range: "Stable relative to cost of revenue is healthier.",
  },
  {
    match: /other current liabilities/i,
    definition: "Short-term obligations not classified elsewhere.",
    calc: "Company-reported current liabilities bucket.",
    range: "Lower relative to current assets is healthier.",
  },
  {
    match: /other non current assets/i,
    definition: "Long-term assets not classified elsewhere.",
    calc: "Company-reported long-term assets bucket.",
    range: "Stable relative to total assets is healthier.",
  },
  {
    match: /total non current assets/i,
    definition: "Assets not expected to be converted within a year.",
    calc: "Total assets - current assets.",
    range: "Balanced vs current assets is healthier.",
  },
  {
    match: /accrued|accrued expenses/i,
    definition: "Expenses incurred but not yet paid.",
    calc: "Recognized expenses awaiting payment.",
    range: "Stable relative to expenses is healthier.",
  },
  {
    match: /deferred revenue|current deferred/i,
    definition: "Cash received for services not yet delivered.",
    calc: "Billings received - revenue recognized.",
    range: "Growing with revenue is healthy.",
  },
  {
    match: /current liabilities/i,
    definition: "Obligations due within a year.",
    calc: "Short-term debt + payables + accruals + deferred revenue.",
    range: "Lower relative to current assets is healthier.",
  },
  {
    match: /income tax payable|total tax payable/i,
    definition: "Taxes owed but not yet paid.",
    calc: "Accrued income taxes payable.",
    range: "Stable relative to pre-tax income is healthier.",
  },
  {
    match: /short term debt|current debt/i,
    definition: "Debt due within the next year.",
    calc: "Portion of borrowings due within 12 months.",
    range: "Lower relative to cash flow is healthier.",
  },
  {
    match: /commercial paper/i,
    definition: "Short-term unsecured borrowing.",
    calc: "Outstanding commercial paper issued.",
    range: "Lower relative to cash is healthier.",
  },
  {
    match: /other current borrowings/i,
    definition: "Other short-term borrowings.",
    calc: "Company-reported short-term debt line item.",
    range: "Lower relative to cash flow is healthier.",
  },
  {
    match: /long term debt and capital lease obligation|long term debt/i,
    definition: "Debt due beyond one year.",
    calc: "Borrowings with maturities over 12 months.",
    range: "Lower relative to equity and cash flow is healthier.",
  },
  {
    match: /total debt/i,
    definition: "All interest-bearing debt outstanding.",
    calc: "Short-term debt + long-term debt.",
    range: "Lower relative to EBITDA is healthier.",
  },
  {
    match: /net debt/i,
    definition: "Debt after subtracting cash on hand.",
    calc: "Total debt - cash and cash equivalents.",
    range: "Lower is healthier; negative is net cash.",
  },
  {
    match: /lease obligation|capital lease obligation/i,
    definition: "Present value of lease payments owed.",
    calc: "Discounted future lease payments.",
    range: "Lower relative to cash flow is healthier.",
  },
  {
    match: /total non current liabilities|other non current liabilities/i,
    definition: "Long-term obligations not due within a year.",
    calc: "Long-term debt + leases + other long-term liabilities.",
    range: "Lower relative to assets is healthier.",
  },
  {
    match: /total liabilities/i,
    definition: "All debts and obligations owed.",
    calc: "Current liabilities + non-current liabilities.",
    range: "Lower relative to assets is healthier.",
  },
  {
    match: /total equity gross minority interest/i,
    definition: "Total equity including minority interest.",
    calc: "Stockholders equity + minority interest.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /minority interest/i,
    definition: "Equity portion owned by non-controlling shareholders.",
    calc: "Subsidiary equity not attributable to the parent.",
    range: "Smaller relative to total equity is typical.",
  },
  {
    match: /stockholders equity|shareholders equity|common stock equity/i,
    definition: "Net value attributable to shareholders.",
    calc: "Total assets - total liabilities.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /total equity\b/i,
    definition: "Net value attributable to equity holders.",
    calc: "Total assets - total liabilities.",
    range: "Positive and growing is healthier.",
  },
  {
    match: /retained earnings/i,
    definition: "Cumulative earnings kept in the business.",
    calc: "Prior retained earnings + net income - dividends.",
    range: "Positive and rising is healthier.",
  },
  {
    match: /capital stock|common stock/i,
    definition: "Par or stated value of issued shares.",
    calc: "Par value x shares issued (company-reported).",
    range: "N/A; structural equity line item.",
  },
  {
    match: /treasury shares|treasury stock/i,
    definition: "Shares repurchased and held by the company.",
    calc: "Cumulative buybacks not retired.",
    range: "Declining share count can be shareholder-friendly.",
  },
  {
    match: /ordinary shares number|share issued|shares issued/i,
    definition: "Total shares issued by the company.",
    calc: "Company-reported issued share count.",
    range: "Stable or declining is typically healthier.",
  },
  {
    match: /working capital/i,
    definition: "Short-term liquidity buffer.",
    calc: "Current assets - current liabilities.",
    range: "Positive is generally healthier.",
  },
  {
    match: /tangible book value|net tangible assets/i,
    definition: "Equity excluding goodwill and intangibles.",
    calc: "Shareholder equity - goodwill - intangibles.",
    range: "Positive and rising is healthier.",
  },
  {
    match: /invested capital/i,
    definition: "Capital invested by debt and equity holders.",
    calc: "Total debt + equity - excess cash.",
    range: "Stable or rising with returns is healthier.",
  },
  {
    match: /total capitalization/i,
    definition: "Long-term debt plus equity.",
    calc: "Long-term debt + shareholder equity.",
    range: "Balanced mix relative to peers is healthier.",
  },
  {
    match: /operating cash flow|cash from operating activities/i,
    definition: "Cash generated by core operations.",
    calc: "Net income + non-cash items +/- working capital changes.",
    range: "Positive and rising is healthy.",
  },
  {
    match: /cash from investing activities/i,
    definition: "Cash used for investments and asset purchases.",
    calc: "Capex + acquisitions + investment sales.",
    range: "Negative can be healthy if investing for growth.",
  },
  {
    match: /cash from financing activities/i,
    definition: "Cash from debt, equity, and shareholder returns.",
    calc: "Debt issuance/repayment + dividends + buybacks.",
    range: "Varies; check consistency with strategy.",
  },
  {
    match: /capital expenditures|capex/i,
    definition: "Cash spent on long-term assets.",
    calc: "Cash paid for PP&E and capitalized software.",
    range: "Sustainable vs operating cash flow is healthy.",
  },
  {
    match: /free cash flow/i,
    definition: "Cash available after capex.",
    calc: "Operating cash flow - capital expenditures.",
    range: "Positive and growing is healthy.",
  },
  {
    match: /dividends paid/i,
    definition: "Cash returned to shareholders via dividends.",
    calc: "Cash dividends paid during the period.",
    range: "Sustainable vs free cash flow is healthy.",
  },
  {
    match: /share repurchase|repurchase of stock|common stock repurchased|buyback/i,
    definition: "Cash used to repurchase shares.",
    calc: "Cash outflow for buybacks.",
    range: "Healthy if supported by strong free cash flow.",
  },
  {
    match: /issuance of stock|common stock issued/i,
    definition: "Cash raised by issuing shares.",
    calc: "Proceeds from equity issuance.",
    range: "Low or stable dilution is healthier.",
  },
  {
    match: /issuance of debt|repayment of debt|net borrowings/i,
    definition: "Net cash flow from debt financing.",
    calc: "Debt issued - debt repaid.",
    range: "Stable leverage is healthier.",
  },
  {
    match: /change in working capital/i,
    definition: "Working capital movement affecting cash flow.",
    calc: "Increase/decrease in current assets and liabilities.",
    range: "Stable and manageable swings are healthier.",
  },
  {
    match: /change in .*receivable/i,
    definition: "Movement in accounts receivable.",
    calc: "Current receivables - prior period receivables.",
    range: "Smaller increases are healthier.",
  },
  {
    match: /change in .*inventory/i,
    definition: "Movement in inventory levels.",
    calc: "Current inventory - prior period inventory.",
    range: "Smaller increases are healthier.",
  },
  {
    match: /change in .*payable/i,
    definition: "Movement in accounts payable.",
    calc: "Current payables - prior period payables.",
    range: "Stable relative to revenue is healthier.",
  },
  {
    match: /change in .*deferred revenue/i,
    definition: "Movement in deferred revenue balance.",
    calc: "Current deferred revenue - prior period deferred revenue.",
    range: "Stable relative to revenue is healthier.",
  },
  {
    match: /effect of exchange rate/i,
    definition: "Impact of FX changes on cash.",
    calc: "Translation effect on cash balances.",
    range: "Smaller impact is healthier.",
  },
];

function compactCurrency(value, currency = "USD") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatCurrency(value, currency = "USD") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(Number(value));
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const num = Number(value);
  const ratio = Math.abs(num) <= 1 ? num * 100 : num;
  return `${ratio.toFixed(2)}%`;
}

function formatPercentRaw(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toFixed(2)}%`;
}

function formatText(value) {
  const text = value === null || value === undefined ? "" : String(value).trim();
  return text || "--";
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function compactNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(
    Number(value),
  );
}

function formatFundamental({ type, value, currency }) {
  switch (type) {
    case "currency":
      return formatCurrency(value, currency);
    case "compactCurrency":
      return compactCurrency(value, currency);
    case "percent":
      return formatPercent(value);
    case "compactNumber":
      return compactNumber(value);
    case "text":
      return formatText(value);
    default:
      return formatNumber(value);
  }
}

function formatTableValue(value) {
  if (value === null || value === undefined || value === "") return "--";
  const num = Number(value);
  if (Number.isFinite(num)) {
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return String(value);
}

function formatRecordValue(value, column, currency = "USD") {
  if (value === null || value === undefined || value === "") return "--";
  const text = String(value);
  if (text.includes("T") || /^\d{4}-\d{2}-\d{2}/.test(text)) {
    const dt = new Date(text);
    if (!Number.isNaN(dt.valueOf())) {
      return dt.toLocaleString();
    }
  }
  const num = Number(value);
  if (Number.isFinite(num)) {
    const col = String(column || "").toLowerCase();
    if (col.includes("earnings move %")) {
      return formatPercentRaw(num);
    }
    if (col.includes("%") || col.includes("percent") || col.includes("surprise")) {
      return formatPercent(num);
    }
    if (col.includes("$") || col.includes("amount")) {
      return formatCurrency(num, currency);
    }
    return num.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return text;
}

function getMetricInfo(key) {
  return METRIC_INFO[key] || null;
}

function getStatementInfo(label) {
  const text = String(label || "").trim();
  if (!text) {
    return {
      definition: "Company-reported financial statement line item.",
      calc: "Defined by company reporting; check footnotes for exact method.",
      range: "Compare trends over time and against peers.",
    };
  }
  const match = STATEMENT_ITEM_HINTS.find((item) => item.match.test(text));
  if (match) return { definition: match.definition, calc: match.calc, range: match.range };
  return {
    definition: "Company-reported financial statement line item.",
    calc: "Defined by company reporting; check footnotes for exact method.",
    range: "Compare trends over time and against peers.",
  };
}

function InfoTooltip({ label, info }) {
  if (!info) {
    return <span>{label}</span>;
  }
  return (
    <Tooltip
      arrow
      placement="top-start"
      title={
        <Box sx={{ maxWidth: 260 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            {label}
          </Typography>
          <Typography variant="body2">{info.definition}</Typography>
          {info.calc ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
              Calculated as: {info.calc}
            </Typography>
          ) : null}
          <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
            Healthy range: {info.range}
          </Typography>
        </Box>
      }
    >
      <Box component="span" sx={{ cursor: "help", borderBottom: "1px dotted", borderColor: "text.secondary" }}>
        {label}
      </Box>
    </Tooltip>
  );
}

function KeyValueTable({ rows, currency }) {
  return (
    <Table size="small">
      <TableBody>
        {rows.map((row) => (
          <TableRow key={row.key}>
            <TableCell sx={{ width: "55%" }}>
              <Typography variant="body2" fontWeight={600}>
                <InfoTooltip label={row.label} info={getMetricInfo(row.key)} />
              </Typography>
            </TableCell>
            <TableCell align="right">
              <Typography variant="body2" color="text.primary">
                {formatFundamental({ type: row.type, value: row.value, currency })}
              </Typography>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function StatementTable({ title, payload }) {
  const columns = payload?.columns || [];
  const index = payload?.index || [];
  const data = payload?.data || [];
  return (
    <Paper sx={{ p: 2.5, display: "grid", gap: 1.5 }} variant="outlined">
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
        <Typography variant="subtitle1" fontWeight={700}>
          {title}
        </Typography>
        {columns.length ? <Chip size="small" variant="outlined" label={`${columns.length} periods`} /> : null}
      </Stack>
      <Divider />
      {columns.length && index.length ? (
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ minWidth: 220 }}>Line item</TableCell>
                {columns.map((col) => (
                  <TableCell key={col} align="right" sx={{ minWidth: 140 }}>
                    {col}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {index.map((rowLabel, rowIdx) => (
                <TableRow key={rowLabel}>
                  <TableCell sx={{ fontWeight: 600 }}>
                    <InfoTooltip label={rowLabel} info={getStatementInfo(rowLabel)} />
                  </TableCell>
                  {columns.map((col, colIdx) => (
                    <TableCell key={`${rowLabel}-${col}`} align="right">
                      {formatTableValue(data[rowIdx]?.[colIdx])}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No data available for this statement yet.
        </Typography>
      )}
    </Paper>
  );
}

function RecordsTable({ title, rows, currency }) {
  const columns = useMemo(() => {
    if (!rows.length) return [];
    return Object.keys(rows[0]);
  }, [rows]);

  return (
    <Paper sx={{ p: 2.5, display: "grid", gap: 1.5 }} variant="outlined">
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
        <Typography variant="subtitle1" fontWeight={700}>
          {title}
        </Typography>
        {rows.length ? <Chip size="small" variant="outlined" label={`${rows.length} entries`} /> : null}
      </Stack>
      <Divider />
      {rows.length ? (
        <Box sx={{ overflowX: "auto" }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                {columns.map((col) => (
                  <TableCell key={col} sx={{ minWidth: 140 }}>
                    {col}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((row, idx) => (
                <TableRow key={`${title}-${idx}`}>
                  {columns.map((col) => (
                    <TableCell key={`${title}-${idx}-${col}`}>
                      {formatRecordValue(row[col], col, currency)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      ) : (
        <Typography variant="body2" color="text.secondary">
          No data available yet.
        </Typography>
      )}
    </Paper>
  );
}

export default function TickerFundamentalsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const searchTicker = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get("ticker")?.trim() || "";
  }, [location.search]);

  const [ticker, setTicker] = useState(searchTicker || "AAPL");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchDetails(symbol) {
    const trimmed = symbol.trim();
    if (!trimmed) {
      setError("Enter a ticker to search.");
      setData(null);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch(
        `${API_BASE}/api/ticker/fundamentals-detail?ticker=${encodeURIComponent(trimmed)}`,
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const body = await res.json();
      setData(body);
    } catch (err) {
      setError(err.message || "Failed to load fundamentals.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (searchTicker) {
      setTicker(searchTicker);
      fetchDetails(searchTicker);
      return;
    }
    fetchDetails(ticker);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchTicker]);

  function handleSearch() {
    const trimmed = ticker.trim();
    if (!trimmed) {
      setError("Enter a ticker to search.");
      return;
    }
    const next = `/ticker/fundamentals?ticker=${encodeURIComponent(trimmed)}`;
    if (searchTicker && searchTicker.toUpperCase() === trimmed.toUpperCase()) {
      fetchDetails(trimmed);
      return;
    }
    navigate(next);
  }

  const fundamentals = data?.fundamentals || {};
  const tradingInfo = data?.trading_info || {};
  const currency = fundamentals?.currency || tradingInfo?.currency || "USD";
  const displayTicker = (data?.ticker || ticker || "").toUpperCase();
  const displayName = (fundamentals?.name || "").trim();
  const titleLabel = displayName ? `${displayName} (${displayTicker})` : displayTicker || "Ticker fundamentals";
  const backLink = displayTicker ? `/ticker?ticker=${encodeURIComponent(displayTicker)}` : "/ticker";
  const fundamentalsRows = FUNDAMENTAL_FIELDS.map((field) => ({
    ...field,
    value: fundamentals?.[field.key],
  }));
  const tradingRows = TRADING_FIELDS.map((field) => ({
    ...field,
    value: tradingInfo?.[field.key],
  }));
  const earningsRows = Array.isArray(data?.earnings_dates) ? data.earnings_dates : [];
  const sharesRows = Array.isArray(data?.shares_history) ? data.shares_history : [];

  const earningsDateKey = useMemo(() => {
    if (!earningsRows.length) return "";
    const keys = Object.keys(earningsRows[0]);
    return (
      keys.find((key) => key.toLowerCase().includes("earnings date")) ||
      keys.find((key) => key.toLowerCase().includes("date")) ||
      keys[0]
    );
  }, [earningsRows]);

  const upcomingEarnings = useMemo(() => {
    if (!earningsDateKey) return null;
    const now = new Date();
    const upcoming = earningsRows
      .map((row) => ({ date: new Date(row[earningsDateKey] || ""), row }))
      .filter((entry) => !Number.isNaN(entry.date.valueOf()) && entry.date >= now)
      .sort((a, b) => a.date - b.date);
    return upcoming[0] || null;
  }, [earningsDateKey, earningsRows]);

  const recentEarnings = useMemo(() => {
    if (!earningsDateKey) return null;
    const now = new Date();
    const past = earningsRows
      .map((row) => ({ date: new Date(row[earningsDateKey] || ""), row }))
      .filter((entry) => !Number.isNaN(entry.date.valueOf()) && entry.date <= now)
      .sort((a, b) => b.date - a.date);
    return past[0] || null;
  }, [earningsDateKey, earningsRows]);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Stack spacing={2.5}>
        <Paper sx={{ p: 2.5, display: "grid", gap: 1.5 }} variant="outlined">
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Link component={RouterLink} to={backLink} underline="hover" sx={{ display: "inline-flex", gap: 0.5 }}>
              <ArrowBack fontSize="small" />
              Back to Ticker Analysis
            </Link>
            {data?.fundamentals_source ? (
              <Chip size="small" variant="outlined" label={`Fundamentals: ${data.fundamentals_source}`} />
            ) : null}
          </Stack>
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems={{ xs: "stretch", md: "center" }}
            justifyContent="space-between"
          >
            <Box>
              <Typography variant="h4" fontWeight={700}>
                Fundamentals Detail
              </Typography>
              <Typography variant="subtitle1" color="text.secondary">
                {titleLabel}
              </Typography>
            </Box>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems="center">
              <TextField
                label="Ticker"
                size="small"
                value={ticker}
                onChange={(event) => setTicker(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    handleSearch();
                  }
                }}
              />
              <Button variant="contained" startIcon={<Search />} onClick={handleSearch}>
                Load
              </Button>
            </Stack>
          </Stack>
        </Paper>

        {error ? <Alert severity="warning">{error}</Alert> : null}
        {loading ? (
          <Box sx={{ display: "grid", placeItems: "center", py: 6 }}>
            <CircularProgress />
          </Box>
        ) : !data ? (
          <Alert severity="info">Search a ticker to load detailed fundamentals.</Alert>
        ) : (
          <Stack spacing={2.5}>
            <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
              <Paper sx={{ p: 2.5, flex: 2 }} variant="outlined">
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                  <Typography variant="subtitle1" fontWeight={700}>
                    Fundamental snapshot
                  </Typography>
                  {displayTicker ? <Chip size="small" label={displayTicker} /> : null}
                </Stack>
                <Divider sx={{ my: 1.5 }} />
                <KeyValueTable rows={fundamentalsRows} currency={currency} />
              </Paper>
              <Paper sx={{ p: 2.5, flex: 2 }} variant="outlined">
                <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
                  <Typography variant="subtitle1" fontWeight={700}>
                    Trading snapshot
                  </Typography>
                  {tradingInfo?.exchange ? (
                    <Chip size="small" variant="outlined" label={tradingInfo.exchange} />
                  ) : null}
                </Stack>
                <Divider sx={{ my: 1.5 }} />
                <KeyValueTable rows={tradingRows} currency={currency} />
              </Paper>
              <Paper sx={{ p: 2.5, flex: 1, display: "grid", gap: 1.5 }} variant="outlined">
                <Typography variant="subtitle1" fontWeight={700}>
                  Important dates
                </Typography>
                {upcomingEarnings ? (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Next earnings
                    </Typography>
                    <Typography variant="body1" fontWeight={600}>
                      {upcomingEarnings.date.toLocaleString()}
                    </Typography>
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No upcoming earnings on file yet.
                  </Typography>
                )}
                {recentEarnings ? (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Most recent earnings
                    </Typography>
                    <Typography variant="body2">
                      {recentEarnings.date.toLocaleString()}
                    </Typography>
                  </Box>
                ) : null}
              </Paper>
            </Stack>

            <RecordsTable title="Earnings dates" rows={earningsRows} currency={currency} />
            <RecordsTable title="Shares outstanding history" rows={sharesRows} currency={currency} />

            <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
              <StatementTable title="Balance sheet (annual)" payload={data?.balance_sheet?.annual} />
              <StatementTable title="Balance sheet (quarterly)" payload={data?.balance_sheet?.quarterly} />
            </Stack>
            <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
              <StatementTable title="Income statement (annual)" payload={data?.income_statement?.annual} />
              <StatementTable title="Income statement (quarterly)" payload={data?.income_statement?.quarterly} />
            </Stack>
            <Stack direction={{ xs: "column", lg: "row" }} spacing={2} alignItems="stretch">
              <StatementTable title="Cashflow (annual)" payload={data?.cashflow?.annual} />
              <StatementTable title="Cashflow (quarterly)" payload={data?.cashflow?.quarterly} />
            </Stack>
          </Stack>
        )}
      </Stack>
    </Container>
  );
}
