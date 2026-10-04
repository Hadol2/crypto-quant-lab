from typing import Optional
import os
import re
import requests
from datetime import datetime
from io import StringIO


# ====== ENV ======
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
FRED_API_KEY = os.environ["FRED_API_KEY"]

NOTION_VERSION = "2022-06-28"

# ====== Notion API ======
NOTION_PAGES_URL = "https://api.notion.com/v1/pages"
NOTION_DATABASE_QUERY_URL = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"

def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }

def notion_create_row(props: dict) -> str:
    payload = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": props}
    r = requests.post(NOTION_PAGES_URL, headers=notion_headers(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def notion_update_row(page_id: str, props: dict) -> str:
    url = f"{NOTION_PAGES_URL}/{page_id}"
    payload = {"properties": props}
    r = requests.patch(url, headers=notion_headers(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]

def notion_find_row_by_date(date_iso: str) -> Optional[str]:
    """
    Find an existing row where property 'Date' equals date_iso (YYYY-MM-DD).
    Returns page_id or None.
    """
    payload = {
        "filter": {
            "property": "Date",
            "date": {"equals": date_iso}
        },
        "page_size": 1
    }
    r = requests.post(NOTION_DATABASE_QUERY_URL, headers=notion_headers(), json=payload, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    return results[0]["id"]

# ====== Data Sources ======
def btc_usd_and_7d_range():
    """
    CryptoCompare free endpoints (no key required in most cases).
    - current price: /data/price
    - 7D hourly candles: /data/v2/histohour?limit=168
    """
    # current
    r = requests.get(
        "https://min-api.cryptocompare.com/data/price",
        params={"fsym": "BTC", "tsyms": "USD"},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    price = float(r.json()["USD"])

    # last 7 days hourly (168 hours)
    r2 = requests.get(
        "https://min-api.cryptocompare.com/data/v2/histohour",
        params={"fsym": "BTC", "tsym": "USD", "limit": 168},
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r2.raise_for_status()
    data = r2.json()["Data"]["Data"]
    low_7d = min(float(c["low"]) for c in data)
    high_7d = max(float(c["high"]) for c in data)
    return price, low_7d, high_7d

def fred_latest(series_id: str) -> float:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 10,
        },
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    for o in r.json().get("observations", []):
        v = o.get("value")
        if v not in (None, "", "."):
            return float(v)
    raise RuntimeError(f"No numeric obs for {series_id}")

def bitbo_etf_10d_net_usdm() -> float:
    """
    Robust-ish Bitbo ETF flows parser:
    - Reads HTML tables and tries multiple strategies:
      1) If a 'Total' row exists: sum last 10 numeric cells in that row
      2) Else if a 'Total' column exists (per-day totals): sum last 10 rows of that column
      3) Else: sum last 10 rows across all fund columns (IBIT/FBTC/GBTC...) to reconstruct daily total
    Returns: last-10-trading-days net flow total in USDm
    """
    import pandas as pd

    url = "https://bitbo.io/treasuries/etf-flows/"
    html = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text

    # read all tables
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("No HTML tables found on Bitbo ETF flows page.")

    # pick the most likely table: contains common ETF tickers
    tickers = {"IBIT", "FBTC", "GBTC", "ARKB", "BITB", "HODL", "BTCO", "BRRR", "EZBC"}
    target = None
    for t in tables:
        cols = {str(c).upper() for c in t.columns}
        # sometimes tickers are in first column values, not headers
        first_col_vals = set(map(lambda x: str(x).upper(), t.iloc[:, 0].dropna().astype(str).tolist())) if t.shape[1] > 0 else set()
        if cols & tickers or first_col_vals & tickers:
            target = t
            break

    # fallback: biggest table
    if target is None:
        target = max(tables, key=lambda x: x.shape[0] * x.shape[1])

    df = target.copy()

    # Normalize: make everything string first
    df.columns = [str(c) for c in df.columns]

    # Helper: convert to numeric safely
    def to_num(s):
        if s is None:
            return None
        s = str(s).strip()
        # remove commas, $ and extra symbols
        s = s.replace(",", "").replace("$", "")
        # handle blanks
        if s in ("", "—", "-", "nan", "NaN"):
            return None
        try:
            return float(s)
        except:
            return None

    # --- Strategy 1: 'Total' row exists ---
    # Look for row label 'Total' in first column
    first_col_name = df.columns[0]
    labels = df[first_col_name].astype(str).str.strip().str.lower()
    if (labels == "total").any():
        total_row = df[labels == "total"].iloc[0]
        nums = [to_num(x) for x in total_row.tolist()[1:]]  # skip label column
        nums = [x for x in nums if x is not None]
        if len(nums) >= 10:
            return float(sum(nums[-10:]))
        # if only one 'total' number exists
        if len(nums) == 1:
            return float(nums[0])

    # --- Strategy 2: 'Total' column exists ---
    total_cols = [c for c in df.columns if c.strip().lower() in ("total", "net", "net flow", "netflows", "net_flows")]
    if total_cols:
        c = total_cols[0]
        series = [to_num(x) for x in df[c].tolist()]
        series = [x for x in series if x is not None]
        if len(series) >= 10:
            return float(sum(series[-10:]))

    # --- Strategy 3: reconstruct daily totals by summing ETF columns row-wise ---
    # Identify numeric columns except the date/label column.
    # Keep columns that look like tickers OR numeric-heavy columns.
    numeric_cols = []
    for c in df.columns[1:]:
        vals = [to_num(x) for x in df[c].tolist()]
        nonnull = [v for v in vals if v is not None]
        # treat as numeric column if enough numeric entries
        if len(nonnull) >= max(3, int(0.2 * len(df))):
            numeric_cols.append(c)

    if not numeric_cols:
        # As a debug help, print columns so user can adjust
        print("❌ Bitbo parse failed. Columns found:", df.columns.tolist())
        print(df.head(5))
        raise RuntimeError("Could not identify numeric columns for ETF flow totals.")

    # compute per-row total across numeric columns
    row_totals = []
    for i in range(len(df)):
        s = 0.0
        ok = False
        for c in numeric_cols:
            v = to_num(df.at[i, c])
            if v is not None:
                s += v
                ok = True
        if ok:
            row_totals.append(s)

    if len(row_totals) >= 10:
        return float(sum(row_totals[-10:]))

    raise RuntimeError("Not enough data points to compute 10-day total from Bitbo table.")
# ====== Main ======
def main():
    today = datetime.now().date().isoformat()

    # 1) Gather data
    btc_price, btc_low, btc_high = btc_usd_and_7d_range()
    ndx = fred_latest("NASDAQ100")
    vxn = fred_latest("VXNCLS")

    # ETF 10D Net from Bitbo (fallback to 0.0 if parsing fails)
    try:
        etf_10d_net_usdm = bitbo_etf_10d_net_usdm()
        notes = f"auto (bitbo ok: {etf_10d_net_usdm:.2f})"
    except Exception as e:
        etf_10d_net_usdm = 0.0
        notes = f"auto (bitbo fail: {type(e).__name__})"
        print("⚠️ Bitbo parse failed; defaulting ETF 10D Net to 0.0:", repr(e))

    props = {
        "Date": {"date": {"start": today}},
        "BTC Price": {"number": round(btc_price, 2)},
        "BTC 7D Low": {"number": round(btc_low, 2)},
        "BTC 7D High": {"number": round(btc_high, 2)},
        "NDX Close": {"number": round(ndx, 2)},
        "VXN Close": {"number": round(vxn, 2)},
        "ETF 10D Net (USDm)": {"number": round(etf_10d_net_usdm, 2)},
        "Notes": {"rich_text": [{"text": {"content": notes}}]},
    }

    # 2) Upsert by date (avoid duplicates)
    existing_page_id = notion_find_row_by_date(today)
    if existing_page_id:
        updated_id = notion_update_row(existing_page_id, props)
        print("✅ updated row:", updated_id)
    else:
        created_id = notion_create_row(props)
        print("✅ created row:", created_id)

if __name__ == "__main__":
    main()