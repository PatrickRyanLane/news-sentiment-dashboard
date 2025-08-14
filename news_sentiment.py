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
NGRAM_RANGE = (2, 3)             # bigrams/trigrams for clearer themes

# Optional: domains to skip (press-release wires, etc.)
BLOCKED_DOMAINS = {
    # "www.prnewswire.com",
    # "www.businesswire.com",
}
# ------------------------------------------


def read_brands_txt():
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
