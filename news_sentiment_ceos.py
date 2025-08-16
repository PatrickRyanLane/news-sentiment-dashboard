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

Run locally or via GitHub Actions. Safe to re-run for the same day; it overwrites
that day's per-article file and updates the daily_counts row(s).
"""

import os
import csv
import time
import math
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
    "www.prnewswire.c
