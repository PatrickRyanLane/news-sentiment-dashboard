#!/usr/bin/env python3
"""
Simplified CEO News Sentiment:

- Input: ceo_aliases.csv with two columns:
    brand,alias        # brand = CEO name, alias = Company name
  (exactly one row per CEO)

- Output:
    data_ceos/articles/YYYY-MM-DD.csv  (per-article rows)
    data_ceos/daily_counts.csv         (aggregated daily totals incl. company + theme)

- Behavior:
    * Query Google News RSS with:  "<CEO NAME>" "<COMPANY>"
    * Cap headlines per CEO per day: MAX_ITEMS_PER_QUERY
    * Pause between CEOs to be polite: REQUEST_PAUSE_SEC
    * Purge files/rows older than PURGE_OLDER_THAN_DAYS
    * Compute a short "theme" from negative headlines (bigrams/trigrams)

Safe to re-run: overwrites today's per-article CSV and updates/merges today's row in daily_counts.csv.
"""

import os, csv, time, math, re
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from urllib.parse import urlencode, urlparse

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# ------------------ Config ------------------
HL   = "en-US"      # interface language
GL   = "US"         # geography
CEID = "US:en"      # country:lang

ALIASES_CSV = "ceo_aliases.csv"   # brand,alias (CEO, Company)

OUTPUT_BASE  = "data_ceos"
ARTICLES_DIR = os.path.join(OUTPUT_BASE, "articles")
COUNTS_CSV   = os.path.join(OUTPUT_BASE, "daily_counts.csv")

REQUEST_PAUSE_SEC     = 1.0  # polite delay between CEOs
MAX_ITEMS_PER_QUERY   = 40   # cap per CEO per day
PURGE_OLDER_THAN_DAYS = 90

# Theme extraction
MIN_THEME_FREQ   = 2
NGRAM_RANGE      = (2, 3)    # bigrams/trigrams
MAX_THEME_WORDS  = 10
STOPWORDS = set((
    "the","a","an","and","or","but","of","for","to","in","on","at","by","with","from","as","about","after","over","under",
    "this","that","these","those","it","its","their","his","her","they","we","you","our","your","i",
    "is","are","was","were","be","been","being","has","have","had","do","does","did","will","would","should","can","could","may","might","must",
    "new","update","updates","report","reports","reported","says","say","said","see","sees","seen","watch","market","stock","shares","share","price","prices",
    "wins","loss","losses","gain","gains","up","down","amid","amidst","news","today","latest","analyst","analysts","rating","cut","cuts","downgrade","downgrades",
    "quarter","q1","q2","q3","q4","year","yrs","2024","2025","2026","usd","billion","million","percent","pct","vs","inc","corp","co","ltd","plc"
))

# Filter out press-release heavy / low-signal sources (tweak as desired)
BLOCKED_DOMAINS = {
    "www.prnewswire.com",
    "www.businesswire.com",
    "www.globenewswire.com",
    "investorplace.com",
    "seekingalpha.com",
}

# --------------- Helpers ----------------

def today_str() -> str:
    return datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")

def load_ceo_company_map(path: str) -> dict:
    """
    Read ceo_aliases.csv (brand,alias) where alias = company name.
    Returns: { CEO -> Company }
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ceo = (row.get("brand") or "").strip()
            company = (row.get("alias") or "").strip()
            if ceo and company:
                out[ceo] = company
    return out

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

def fetch_items_for_query(query: str, cap: int) -> list[dict]:
    """
    Return up to cap items: {title, link, published, domain}
    """
    url = google_news_rss_url(query)
    parsed = feedparser.parse(url)
    out = []
    for e in parsed.entries[: cap]:
        link = e.get("link") or ""
        dom = domain_of(link)
        if dom in BLOCKED_DOMAINS:
            continue
        out.append({
            "title": (e.get("title") or "").strip(),
            "link":  link,
            "published": (e.get("published") or e.get("updated") or "").strip(),
            "domain": dom,
        })
    return out

_analyzer = SentimentIntensityAnalyzer()

def label_sentiment(title: str) -> str:
    s = _analyzer.polarity_scores(title or "")
    v = s["compound"]
    return "positive" if v >= 0.2 else ("negative" if v <= -0.2 else "neutral")

def dedup_by_title_domain(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for x in items:
        key = (x["title"], x["domain"])
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

def clean_for_theme(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def short_theme_from_negatives(neg_titles: list[str]) -> str:
    """
    Build a concise theme from negative headlines.
    Returns <=10 words or 'None' if nothing obvious.
    """
    if not neg_titles:
        return "None"

    docs = [clean_for_theme(t) for t in neg_titles if t.strip()]
    if not docs:
        return "None"

    vec = CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_THEME_FREQ
    )
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return "None"

    counts = X.sum(axis=0).A1
    vocab  = vec.get_feature_names_out()
    if not len(counts):
        return "None"

    top_idx = counts.argmax()
    phrase  = vocab[top_idx]
    words   = phrase.split()
    if len(words) > MAX_THEME_WORDS:
        words = words[:MAX_THEME_WORDS]
    return " ".join(words) if words else "None"

def ensure_dirs():
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

def purge_old_files():
    """
    Delete per-day article CSVs older than cutoff,
    and trim old rows out of data_ceos/daily_counts.csv.
    Use **date objects** to avoid timezone aware/naive comparisons.
    """
    cutoff_date = date.today() - timedelta(days=PURGE_OLDER_THAN_DAYS)

    # Per-day article files
    for name in os.listdir(ARTICLES_DIR):
        if not name.endswith(".csv"):
            continue
        try:
            dt = datetime.strptime(name.replace(".csv", ""), "%Y-%m-%d").date()
        except Exception:
            continue
        if dt < cutoff_date:
            try:
                os.remove(os.path.join(ARTICLES_DIR, name))
            except Exception:
                pass

    # Trim counts
    if os.path.exists(COUNTS_CSV):
        df = pd.read_csv(COUNTS_CSV)
        def keep(row):
            try:
                d = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
                return d >= cutoff_date
            except Exception:
                return True
        if not df.empty:
            df2 = df[df.apply(keep, axis=1)]
            if len(df2) != len(df):
                df2.to_csv(COUNTS_CSV, index=False)

# --------------- Main ----------------

def main():
    print("=== CEO Sentiment (simple alias) : start ===")
    ensure_dirs()
    purge_old_files()

    ceo_to_company = load_ceo_company_map(ALIASES_CSV)
    if not ceo_to_company:
        raise SystemExit(f"No rows found in {ALIASES_CSV}. Expected header 'brand,alias' with alias=Company.")

    today = today_str()
    daily_articles_path = os.path.join(ARTICLES_DIR, f"{today}.csv")

    # Collect per-article rows for today (overwrite the file)
    article_rows = []
    counts_rows = []

    for idx, (ceo, company) in enumerate(ceo_to_company.items(), start=1):
        # Build disambiguated query
        q = f"\"{ceo}\" \"{company}\""
        items = fetch_items_for_query(q, MAX_ITEMS_PER_QUERY)
        items = dedup_by_title_domain(items)

        pos = neu = neg = 0
        neg_titles = []

        for it in items:
            sent = label_sentiment(it["title"])
            if sent == "positive": pos += 1
            elif sent == "negative":
                neg += 1
                neg_titles.append(it["title"])
            else:
                neu += 1

            article_rows.append({
                "date": today,
                "brand": ceo,
                "company": company,
                "title": it["title"],
                "url": it["link"],
                "domain": it["domain"],
                "sentiment": sent,
                "published": it["published"],
            })

        total = pos + neu + neg
        theme = short_theme_from_negatives(neg_titles)

        counts_rows.append({
            "date": today,
            "brand": ceo,
            "company": company,
            "total": total,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "theme": theme,
        })

        # polite pause between CEOs
        if idx < len(ceo_to_company):
            time.sleep(REQUEST_PAUSE_SEC)

    # Write per-article
    with open(daily_articles_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date","brand","company","title","url","domain","sentiment","published"
        ])
        w.writeheader()
        for r in article_rows:
            w.writerow(r)

    # Upsert into daily_counts.csv
    # Strategy: if file exists, drop any existing rows for today's date, then append new rows.
    if os.path.exists(COUNTS_CSV):
        df_old = pd.read_csv(COUNTS_CSV)
        df_old = df_old[df_old["date"] != today]
        df_new = pd.DataFrame(counts_rows, columns=[
            "date","brand","company","total","positive","neutral","negative","theme"
        ])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(COUNTS_CSV, index=False)
    else:
        with open(COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "date","brand","company","total","positive","neutral","negative","theme"
            ])
            w.writeheader()
            for r in counts_rows:
                w.writerow(r)

    print(f"Wrote {len(article_rows)} articles -> {daily_articles_path}")
    print(f"Upserted {len(counts_rows)} rows -> {COUNTS_CSV}")
    print("=== CEO Sentiment (simple alias) : done ===")

if __name__ == "__main__":
    main()
