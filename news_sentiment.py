#!/usr/bin/env python3
"""
Daily Google News RSS -> sentiment counts (positive/neutral/negative) per brand
with alias support and a per-brand 'theme' for the day.

Outputs:
- data/articles/YYYY-MM-DD.csv   (item-level rows)
- data/daily_counts.csv          (aggregated daily totals incl. 'theme')

Inputs (repo root):
- brands.txt     (one brand per line; optional if aliases.csv covers everything)
- aliases.csv    (brand,alias rows; optional; merges with brands.txt)
"""

import os
import time
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# --------- Config ---------
HL = "en-US"     # interface language
GL = "US"        # geography
CEID = "US:en"   # country:lang
REQUEST_PAUSE_SEC = 0.3          # gentle delay between RSS fetches
MIN_THEME_FREQ = 2               # min times a phrase must appear to count as a "theme"

# Optional: domains to skip entirely (press-release wires, etc.)
BLOCKED_DOMAINS = {
    # "www.prnewswire.com",
    # "www.businesswire.com",
}
# ------------------------------------------


def read_brands_txt():
    """Return a set of brands from brands.txt if present."""
    path = "brands.txt"
    brands = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    brands.append(s)
    return set(brands)


def read_aliases_csv():
    """Return dict: brand -> list of aliases from aliases.csv (if present)."""
    path = "aliases.csv"
    mapping = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand = (row.get("brand") or "").strip()
                alias = (row.get("alias") or "").strip()
                if not brand or not alias:
                    continue
                mapping.setdefault(brand, []).append(alias)
    return mapping


def compile_brand_aliases():
    """
    Combine brands.txt and aliases.csv into:
      brands: list of canonical brands
      aliases_map: dict brand -> list of aliases (brand name included if no alias)
    """
    aliases_map = read_aliases_csv()
    brands_from_aliases = set(aliases_map.keys())
    brands_from_txt = read_brands_txt()

    brands = sorted(brands_from_aliases | brands_from_txt)
    # If brands.txt has items with no aliases, make the brand name its own alias
    for b in brands:
        if b not in aliases_map:
            aliases_map[b] = [b]
        else:
            # ensure canonical brand name is also searched
            if b not in aliases_map[b]:
                aliases_map[b].append(b)
    return brands, aliases_map


def normalize_url(u: str) -> str:
    """Normalize URL (strip tracking params, fragments)."""
    try:
        p = urlparse(u)
        keep = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            lk = k.lower()
            if lk.startswith("utm_") or lk in {"gclid", "fbclid"}:
                continue
            keep.append((k, v))
        new_q = urlencode(keep, doseq=True)
        p2 = p._replace(query=new_q, fragment="")
        return urlunparse(p2)
    except Exception:
        return u


def build_rss_url(query: str) -> str:
    """Build a Google News RSS URL from a pre-encoded query string."""
    base = "https://news.google.com/rss/search"
    return f"{base}?q={quote(query)}&hl={HL}&gl={GL}&ceid={CEID}"


def classify_sentiment(text: str, analyzer: SentimentIntensityAnalyzer):
    """
    VADER thresholds:
    >  0.05 => positive
    < -0.05 => negative
    else     neutral
    """
    s = analyzer.polarity_scores(text or "")
    compound = s["compound"]
    if compound > 0.05:
        label = "positive"
    elif compound < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound, s


# ---------- Theme extraction (titles-only) ----------
def pick_theme(titles):
    """
    Extract a short 'theme' from a list of TITLES (not snippets) using most
    frequent bigram/trigram of letters-only tokens. Avoids HTML artifacts like
    'font color 6f6f6f'. Returns 'none' if nothing reaches MIN_THEME_FREQ.
    """
    titles = [t for t in titles if isinstance(t, str) and t.strip()]
    if not titles:
        return "none"

    try:
        vec = CountVectorizer(
            stop_words="english",
            ngram_range=(2, 3),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"  # letters only
        )
        X = vec.fit_transform(titles)
        counts = X.toarray().sum(axis=0)
        vocab = {v: k for k, v in vec.vocabulary_.items()}

        # Sort by frequency desc, then by phrase length desc (prefer trigrams if tied)
        phrases = sorted(
            ((vocab[i], int(counts[i])) for i in range(len(counts))),
            key=lambda x: (x[1], len(x[0])),
            reverse=True
        )

        for phrase, freq in phrases:
            if freq >= MIN_THEME_FREQ:
                return phrase
        return "none"
    except Exception:
        return "none"
# ---------- /Theme extraction ----------


def main():
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)

    # Target the previous calendar day in Eastern Time
    target_day = (now.date() - timedelta(days=1))
    start_str = target_day.strftime("%Y-%m-%d")
    end_str = (target_day + timedelta(days=1)).strftime("%Y-%m-%d")

    brands, aliases_map = compile_brand_aliases()
    if not brands:
        # final fallback
        brands = ["Apple", "Amazon", "Walmart", "Ford", "Disney"]
        aliases_map = {b: [b] for b in brands}

    print(f"Running sentiment for {start_str} (US/Eastern) across {len(brands)} brands")

    analyzer = SentimentIntensityAnalyzer()
    all_rows = []
    seen = set()  # (domain, normalized_title)

    for brand in brands:
        aliases = sorted(set(aliases_map.get(brand, [brand])))
        for alias in aliases:
            # Exact phrase + date window
            query = f"\"{alias}\" after:{start_str} before:{end_str}"
            url = build_rss_url(query)
            feed = feedparser.parse(url)
            time.sleep(REQUEST_PAUSE_SEC)

            for e in getattr(feed, "entries", []):
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                summary = (e.get("summary") or "").strip()

                norm_url = normalize_url(link)
                domain = urlparse(norm_url).netloc.lower()

                if domain in BLOCKED_DOMAINS:
                    continue

                # Deduplicate by (domain, title)
                key = (domain, title.lower())
                if key in seen:
                    continue
                seen.add(key)

                # Sentiment on title + snippet (quick read)
                text_for_sent = f"{title} {summary}".strip()
                label, compound, score = classify_sentiment(text_for_sent, analyzer)
                published = e.get("published") or e.get("updated") or ""

                row = {
                    "date": start_str,
                    "brand": brand,
                    "alias_matched": alias,
                    "title": title,
                    "url": norm_url or link,
                    "domain": domain,
                    "published": published,
                    "sentiment": label,
                    "compound": round(compound, 4),
                    "pos": score["pos"],
                    "neu": score["neu"],
                    "neg": score["neg"],
                    # Keep text for potential future use; not used for theme now.
                    "text": text_for_sent,
                }
                all_rows.append(row)

    # Ensure output folders exist
    os.makedirs("data/articles", exist_ok=True)
    detail_path = f"data/articles/{start_str}.csv"

    # Item-level CSV
    columns = [
        "date", "brand", "alias_matched", "title", "url", "domain", "published",
        "sentiment", "compound", "pos", "neu", "neg"
    ]
    pd.DataFrame(all_rows)[columns].to_csv(detail_path, index=False)

    agg_path = "data/daily_counts.csv"

    if all_rows:
        df = pd.DataFrame(all_rows)

        # THEME: use TITLES ONLY to avoid HTML artifacts in snippets
        themes = (
            df.groupby("brand")["title"]
              .apply(lambda s: pick_theme(list(s)))
              .rename("theme")
              .reset_index()
        )

        counts = (
            df.groupby(["date", "brand", "sentiment"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )
        for col in ["positive", "neutral", "negative"]:
            if col not in counts:
                counts[col] = 0
        counts["total"] = counts["positive"] + counts["neutral"] + counts["negative"]
        counts = counts[["date", "brand", "total", "positive", "neutral", "negative"]]

        # Merge in themes
        daily = counts.merge(themes, on="brand", how="left")
        daily["theme"] = daily["theme"].fillna("none")

        # Idempotent append
        if os.path.exists(agg_path):
            existing = pd.read_csv(agg_path)
            existing = existing[existing["date"] != start_str]
            daily_counts = pd.concat([existing, daily], ignore_index=True)
        else:
            daily_counts = daily

        daily_counts = daily_counts.sort_values(["date", "brand"])
        daily_counts.to_csv(agg_path, index=False)
        print(f"Wrote {len(all_rows)} articles to {detail_path}")
        print(f"Updated {agg_path} (with theme column)")
    else:
        # Leave previous daily_counts.csv untouched if nothing was found today
        print("No articles found; leaving existing daily_counts.csv untouched.")
        print(f"Wrote 0 articles to {detail_path}")

if __name__ == "__main__":
    main()
