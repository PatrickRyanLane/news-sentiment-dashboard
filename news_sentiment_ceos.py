#!/usr/bin/env python3
"""
Daily Google News RSS -> sentiment counts (positive/neutral/negative) per CEO
with alias support and a per-CEO 'theme' (summarized from negative headlines).

Outputs:
- data_ceos/articles/YYYY-MM-DD.csv   (item-level rows)
- data_ceos/daily_counts.csv          (aggregated daily totals incl. 'company' and 'theme')

Inputs (repo root):
- ceos.txt             (one CEO per line; optional if ceo_aliases.csv covers all)
- ceo_aliases.csv      (brand,alias rows; optional; merges with ceos.txt)
- ceo_companies.csv    (brand,company rows; optional; enriches output; shown in dashboard)

Run locally or via GitHub Actions. Safe to re-run for the same day; it overwrites
that day's per-article file and upserts the daily_counts row(s).
"""

import os
import csv
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlencode, urlparse

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# ----------------- Config -----------------
HL = "en-US"     # interface language
GL = "US"        # geography
CEID = "US:en"   # country:lang

# CEOs inputs
BRANDS_TXT = "ceos.txt"
ALIASES_CSV = "ceo_aliases.csv"
CEO_COMPANIES_CSV = "ceo_companies.csv"

# Output base folder (isolated from brands pipeline)
OUTPUT_BASE = "data_ceos"
ARTICLES_DIR = os.path.join(OUTPUT_BASE, "articles")
COUNTS_CSV = os.path.join(OUTPUT_BASE, "daily_counts.csv")

# Throttling & limits
REQUEST_PAUSE_SEC = 1.0       # polite delay between RSS requests
MAX_ITEMS_PER_QUERY = 40      # cap per CEO per day
PURGE_OLDER_THAN_DAYS = 90    # purge old rows/files

# Theme extraction knobs
MIN_THEME_FREQ = 2            # min times a phrase must appear to count as a theme
NGRAM_RANGE = (2, 3)          # bigrams/trigrams
MAX_THEME_WORDS = 10          # safety cap for theme length

# Optional: domains to skip (press releases / noise)
BLOCKED_DOMAINS = {
    "www.prnewswire.com",
    "www.businesswire.com",
    "www.globenewswire.com",
    "investorplace.com",
    "seekingalpha.com",
}

# ----------------- IO helpers -----------------
def ensure_dirs():
    os.makedirs(ARTICLES_DIR, exist_ok=True)

def read_ceos_txt(path: str = BRANDS_TXT) -> list[str]:
    out = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    out.append(s)
    return sorted(set(out))

def read_aliases_csv(path: str = ALIASES_CSV) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                b = (row.get("brand") or "").strip()
                a = (row.get("alias") or "").strip()
                if not b or not a:
                    continue
                m.setdefault(b, []).append(a)
    return m

def read_companies_csv(path: str = CEO_COMPANIES_CSV) -> dict[str, str]:
    m: dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                b = (row.get("brand") or "").strip()
                c = (row.get("company") or "").strip()
                if b and c:
                    m[b] = c
    return m

# ----------------- Query building -----------------
def quoted(s: str) -> str:
    s = s.strip()
    return f"\"{s}\"" if s and not (s.startswith('"') and s.endswith('"')) else s

def all_queries_for(brand: str, alias_map: dict[str, list[str]], company: str | None) -> str:
    """
    Keep it simple and precise:
    ("Tim Cook" OR "Timothy Cook") AND (CEO OR "Apple")
    If no company is provided: ("Tim Cook" OR "Timothy Cook") AND CEO
    """
    parts = [quoted(brand)]
    for a in alias_map.get(brand, []):
        if a.strip().lower() != brand.strip().lower():
            parts.append(quoted(a))
    left = "(" + " OR ".join(parts) + ")"
    right_terms = ["CEO"]
    if company:
        right_terms.append(quoted(company))
    right = "(" + " OR ".join(right_terms) + ")"
    return f"{left} {right}"

def google_news_rss_url(query: str) -> str:
    base = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": HL,
        "gl": GL,
        "ceid": CEID,
    }
    return base + "?" + urlencode(params)

def domain_of(link: str) -> str:
    try:
        return urlparse(link).hostname or ""
    except Exception:
        return ""

# ----------------- Fetch & analyze -----------------
def fetch_items_for_query(query: str, cap: int) -> list[dict]:
    url = google_news_rss_url(query)
    parsed = feedparser.parse(url)
    out: list[dict] = []
    for entry in parsed.entries[: cap]:
        link = entry.get("link") or ""
        dom = domain_of(link)
        if dom in BLOCKED_DOMAINS:
            continue
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "link": link,
            "source": dom,
        })
    return out

def get_sentiment_scores(titles: list[str]) -> tuple[int, int, int]:
    analyzer = SentimentIntensityAnalyzer()
    pos = neu = neg = 0
    for t in titles:
        s = analyzer.polarity_scores(t)
        # VADER thresholds
        if s["compound"] >= 0.05:
            pos += 1
        elif s["compound"] <= -0.05:
            neg += 1
        else:
            neu += 1
    return pos, neu, neg

def pick_theme_from_negatives(neg_titles: list[str]) -> str:
    if not neg_titles:
        return "None"
    # Short stoplist additions relevant to CEOs
    stop_extra = {"ceo", "cfo", "coo", "inc", "corp", "company", "reports", "says"}
    try:
        vec = CountVectorizer(
            ngram_range=NGRAM_RANGE,
            stop_words="english",
            min_df=1,
        )
        X = vec.fit_transform([t.lower() for t in neg_titles])
        vocab = vec.get_feature_names_out()
        counts = X.toarray().sum(axis=0)
        pairs = [(vocab[i], int(counts[i])) for i in range(len(vocab))]
        # filter out pieces that are not useful
        pairs = [
            (ph, c) for ph, c in pairs
            if c >= MIN_THEME_FREQ and not any(tok in stop_extra for tok in ph.split())
        ]
        if not pairs:
            return "None"
        pairs.sort(key=lambda x: (-x[1], x[0]))
        theme = pairs[0][0]
        # cap to MAX_THEME_WORDS words
        theme = " ".join(theme.split()[:MAX_THEME_WORDS])
        return theme
    except Exception:
        return "None"

# ----------------- Purge (date-only safe) -----------------
def purge_old_files():
    """Delete old article-day CSVs and trim daily_counts by date (date-only, no tz)."""
    keep_from = (datetime.now(ZoneInfo("US/Eastern")).date()
                 - timedelta(days=PURGE_OLDER_THAN_DAYS))

    # Delete old per-article files
    if os.path.isdir(ARTICLES_DIR):
        for fname in os.listdir(ARTICLES_DIR):
            if not fname.endswith(".csv"):
                continue
            try:
                d = datetime.strptime(fname[:-4], "%Y-%m-%d").date()
            except Exception:
                continue
            if d < keep_from:
                try:
                    os.remove(os.path.join(ARTICLES_DIR, fname))
                except Exception:
                    pass

    # Trim daily_counts.csv
    if os.path.exists(COUNTS_CSV):
        try:
            df = pd.read_csv(COUNTS_CSV)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
                df = df[df["date"].notna()]
                df = df[df["date"] >= keep_from]
                df.sort_values(["date", "brand"], inplace=True)
                df["date"] = df["date"].astype(str)
                df.to_csv(COUNTS_CSV, index=False)
        except Exception:
            pass

# ----------------- Main -----------------
def main():
    tz = ZoneInfo("US/Eastern")
    today = datetime.now(tz).date()
    day_str = today.strftime("%Y-%m-%d")

    ensure_dirs()
    purge_old_files()

    ceos = read_ceos_txt()
    aliases = read_aliases_csv()
    companies = read_companies_csv()

    # If ceos.txt is empty, fall back to aliases keys
    if not ceos:
        ceos = sorted(set(aliases.keys()))

    print(f"=== CEO Sentiment: start ===")
    print(f"  Date: {day_str}")
    print(f"  CEOs: {len(ceos)}")

    # Per-day article output path
    articles_out = os.path.join(ARTICLES_DIR, f"{day_str}.csv")
    articles_rows: list[list[str]] = []
    art_header = ["date", "brand", "company", "title", "link", "source"]

    # Collect daily counts
    daily_rows: list[dict] = []

    for i, brand in enumerate(ceos, 1):
        company = companies.get(brand, "")
        query = all_queries_for(brand, aliases, company)
        print(f"[{i}/{len(ceos)}] {brand}: {query}")

        items = fetch_items_for_query(query, MAX_ITEMS_PER_QUERY)
        # de-duplicate by title
        seen_titles = set()
        dedup = []
        for it in items:
            t = it["title"]
            if t not in seen_titles:
                seen_titles.add(t)
                dedup.append(it)
        items = dedup

        titles = [it["title"] for it in items]
        pos, neu, neg = get_sentiment_scores(titles)
        total = len(titles)
        neg_titles = [it["title"] for it in items][:MAX_ITEMS_PER_QUERY]  # source list anyway
        theme = pick_theme_from_negatives([t for t in titles if t])

        # append articles rows
        for it in items:
            articles_rows.append([day_str, brand, company, it["title"], it["link"], it["source"]])

        daily_rows.append({
            "date": day_str,
            "brand": brand,
            "company": company,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "total": total,
            "theme": theme if theme else "None",
        })

        time.sleep(REQUEST_PAUSE_SEC)

    # Write per-day articles CSV (overwrite)
    with open(articles_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(art_header)
        w.writerows(articles_rows)

    # Upsert into daily_counts.csv (include company column)
    COLS = ["date", "brand", "company", "positive", "neutral", "negative", "total", "theme"]
    if os.path.exists(COUNTS_CSV):
        df = pd.read_csv(COUNTS_CSV)
        for c in COLS:
            if c not in df.columns:
                df[c] = "" if c in ("company", "theme") else 0
        # Remove any existing rows for (date, brand), then append fresh
        key = set((r["date"], r["brand"]) for _, r in df.iterrows())
        fresh = pd.DataFrame(daily_rows)
        df = df[~( (df["date"] == fresh.iloc[0]["date"]) & (df["brand"].isin(list(fresh["brand"]))) )]
        df = pd.concat([df, fresh], ignore_index=True)
        df = df[COLS]
        df.sort_values(["date", "brand"], inplace=True)
        df.to_csv(COUNTS_CSV, index=False)
    else:
        pd.DataFrame(daily_rows, columns=COLS).to_csv(COUNTS_CSV, index=False)

    print("=== CEO Sentiment: done ===")

if __name__ == "__main__":
    main()
