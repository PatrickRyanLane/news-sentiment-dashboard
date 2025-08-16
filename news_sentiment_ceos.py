#!/usr/bin/env python3
"""
Daily Google News RSS -> sentiment counts (positive/neutral/negative) per CEO
with alias support and a per-CEO 'theme' (summarized from negative headlines).

Outputs:
- data_ceos/articles/YYYY-MM-DD.csv   (item-level rows)
- data_ceos/daily_counts.csv          (aggregated daily totals incl. 'theme')

Inputs (repo root):
- ceos.txt          (one CEO per line; optional if ceo_aliases.csv covers all)
- ceo_aliases.csv   (brand,alias rows; optional; merges with ceos.txt)
"""

import os, csv, time, math, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlencode, urlparse

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# ----------------- Config -----------------
HL   = "en-US"      # interface language
GL   = "US"         # geography
CEID = "US:en"      # country:lang

# CEOs inputs (filenames must exist at repo root)
BRANDS_TXT  = "ceos.txt"
ALIASES_CSV = "ceo_aliases.csv"

# Output base folder (isolated from brands pipeline)
OUTPUT_BASE  = "data_ceos"
ARTICLES_DIR = os.path.join(OUTPUT_BASE, "articles")
COUNTS_CSV   = os.path.join(OUTPUT_BASE, "daily_counts.csv")

# Throttling & limits
REQUEST_PAUSE_SEC     = 1.0     # polite delay between RSS requests
MAX_ITEMS_PER_QUERY   = 40      # cap per CEO per day
PURGE_OLDER_THAN_DAYS = 90      # purge old rows/files

# Theme extraction knobs
MIN_THEME_FREQ  = 2            # min times a phrase must appear to count as a theme
NGRAM_RANGE     = (2, 3)       # bigrams/trigrams
MAX_THEME_WORDS = 10

# Optional: domains to skip (press releases / noise)
BLOCKED_DOMAINS = {
    "www.prnewswire.com",
    "www.businesswire.com",
    "www.globenewswire.com",
    "investorplace.com",
    "seekingalpha.com",
}

def log(msg: str):
    print(msg, flush=True)

def ensure_dirs():
    os.makedirs(ARTICLES_DIR, exist_ok=True)

def read_ceos() -> list[str]:
    ceos = []
    if os.path.exists(BRANDS_TXT):
        with open(BRANDS_TXT, "r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if s and not s.startswith("#"):
                    ceos.append(s)
    return list(dict.fromkeys(ceos))  # de-dupe preserve order

def read_aliases() -> dict[str, list[str]]:
    mapping = {}
    if os.path.exists(ALIASES_CSV):
        with open(ALIASES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand = (row.get("brand") or "").strip()
                alias = (row.get("alias") or "").strip()
                if brand and alias:
                    mapping.setdefault(brand, []).append(alias)
    return mapping

def all_queries_for(brand: str, alias_map: dict[str, list[str]]):
    q = [brand]
    for a in alias_map.get(brand, []):
        if a.lower() != brand.lower():
            q.append(a)
    # Combine aliases into OR query (quoted) and add negative company name (CEO only)
    # e.g. ("Tim Cook" OR "Apple CEO")
    parts = [f'"{x}"' if " " in x else x for x in q]
    return "(" + " OR ".join(parts) + ")"

def google_news_rss_url(query: str):
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
    url = google_news_rss_url(query)
    parsed = feedparser.parse(url)
    out = []
    for entry in parsed.entries[: cap]:
        link = entry.get("link") or ""
        dom  = domain_of(link)
        if dom in BLOCKED_DOMAINS:
            continue
        title = entry.get("title") or ""
        # attempt date from published, fall back empty
        published = entry.get("published") or entry.get("updated") or ""
        out.append({
            "title": title,
            "link": link,
            "domain": dom,
            "published": published,
        })
    return out

def simple_sentiment(labeler, text: str) -> str:
    s = labeler.polarity_scores(text or "")
    comp = s["compound"]
    if comp >= 0.2:  return "positive"
    if comp <= -0.2: return "negative"
    return "neutral"

def theme_from_titles(titles: list[str]) -> str:
    if not titles:
        return "None"
    vec = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words="english")
    X = vec.fit_transform(titles)
    freqs = X.toarray().sum(axis=0)
    vocab = vec.get_feature_names_out()
    # pairs (phrase, count) sorted desc
    pairs = sorted([(vocab[i], int(freqs[i])) for i in range(len(vocab))],
                   key=lambda x: x[1], reverse=True)
    for phrase, count in pairs:
        if count < MIN_THEME_FREQ:
            break
        # short, human-ish cap
        words = phrase.split()
        phrase_short = " ".join(words[:MAX_THEME_WORDS])
        return phrase_short
    return "None"

def today_eastern() -> str:
    now = datetime.now(ZoneInfo("US/Eastern"))
    return now.strftime("%Y-%m-%d")

def purge_old_files():
    cutoff = datetime.now(ZoneInfo("US/Eastern")) - timedelta(days=PURGE_OLDER_THAN_DAYS)
    # delete old per-day articles files
    for name in os.listdir(ARTICLES_DIR):
        if not name.endswith(".csv"): 
            continue
        path = os.path.join(ARTICLES_DIR, name)
        try:
            dt = datetime.strptime(name.replace(".csv",""), "%Y-%m-%d")
        except Exception:
            continue
        if dt < cutoff:
            try: os.remove(path)
            except Exception: pass
    # trim daily_counts.csv rows older than cutoff
    if os.path.exists(COUNTS_CSV):
        df = pd.read_csv(COUNTS_CSV)
        def ok(row):
            try:
                d = datetime.strptime(str(row["date"]), "%Y-%m-%d")
                return d >= cutoff
            except Exception:
                return True
        if not df.empty:
            df2 = df[df.apply(ok, axis=1)]
            if len(df2) != len(df):
                df2.to_csv(COUNTS_CSV, index=False)

def write_articles_csv(date: str, rows: list[dict]):
    path = os.path.join(ARTICLES_DIR, f"{date}.csv")
    fieldnames = ["date", "brand", "alias_matched", "title", "url", "domain", "sentiment", "published"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def upsert_daily_counts(date: str, agg_rows: list[dict]):
    # ensure file exists (header)
    need_header = not os.path.exists(COUNTS_CSV)
    if need_header:
        with open(COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date","brand","total","positive","neutral","negative","theme"])
    # Load existing and replace rows for (date, brand)
    df_existing = pd.read_csv(COUNTS_CSV)
    if not df_existing.empty:
        keys = {(str(r["date"]), str(r["brand"])) for _, r in df_existing.iterrows()}
    else:
        keys = set()
    # Remove any rows for this date that we’re about to write
    df_existing = df_existing[df_existing["date"] != date] if not df_existing.empty else df_existing
    df_new = pd.DataFrame(agg_rows, columns=["date","brand","total","positive","neutral","negative","theme"])
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out.to_csv(COUNTS_CSV, index=False)

def main():
    log("=== CEO Sentiment: start ===")
    ensure_dirs()
    purge_old_files()

    date = today_eastern()
    log(f"Run date (US/Eastern): {date}")
    ceos = read_ceos()
    aliases = read_aliases()
    log(f"Loaded CEOs from {BRANDS_TXT}: {len(ceos)}")
    log(f"Loaded aliases from {ALIASES_CSV}: {sum(len(v) for v in aliases.values())}")

    if not ceos and not aliases:
        log("No CEOs or aliases found. Nothing to do.")
        return

    analyzer = SentimentIntensityAnalyzer()

    per_article = []      # rows for data_ceos/articles/YYYY-MM-DD.csv
    per_brand_agg = {}    # brand -> {pos,neu,neg}

    # Build the canonical set of “brands” (CEO display names) to query
    brands = set(ceos) | set(aliases.keys())
    brands = [b for b in brands if b.strip()]
    log(f"Total CEO identities to query: {len(brands)}")

    for i, brand in enumerate(brands, 1):
        q = all_queries_for(brand, aliases)
        log(f"[{i}/{len(brands)}] Query: {q}")
        items = fetch_items_for_query(q, MAX_ITEMS_PER_QUERY)
        log(f"  ↳ fetched {len(items)} items")
        time.sleep(REQUEST_PAUSE_SEC)

        for it in items:
            sent = simple_sentiment(analyzer, it["title"])
            per_article.append({
                "date": date,
                "brand": brand,
                "alias_matched": "",   # (optional) could store which alias matched
                "title": it["title"],
                "url": it["link"],
                "domain": it["domain"],
                "sentiment": sent,
                "published": it["published"],
            })
            agg = per_brand_agg.setdefault(brand, {"positive":0,"neutral":0,"negative":0})
            agg[sent] += 1

    if not per_article:
        log("WARNING: No articles collected for any CEO today.")
    else:
        # compute themes from NEGATIVE titles per brand
        themed = {}
        for brand, counts in per_brand_agg.items():
            neg_titles = [r["title"] for r in per_article if r["brand"] == brand and r["sentiment"] == "negative"]
            themed[brand] = theme_from_titles(neg_titles)

        # write per-day article file
        write_articles_csv(date, per_article)
        log(f"Wrote {len(per_article)} article rows -> {os.path.join(ARTICLES_DIR, f'{date}.csv')}")

        # build daily_counts rows and upsert
        agg_rows = []
        for brand, counts in per_brand_agg.items():
            total = counts["positive"] + counts["neutral"] + counts["negative"]
            agg_rows.append({
                "date": date,
                "brand": brand,
                "total": total,
                "positive": counts["positive"],
                "neutral": counts["neutral"],
                "negative": counts["negative"],
                "theme": themed.get(brand) or "None"
            })
        upsert_daily_counts(date, agg_rows)
        log(f"Upserted {len(agg_rows)} rows -> {COUNTS_CSV}")

    log("=== CEO Sentiment: done ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make failures visible in GH Actions logs
        print("FATAL:", e, file=sys.stderr, flush=True)
        raise
