# Example: Updating daily_brands.yml to use the composite action

## Before (current version):

```yaml
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.11"
    

- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

## After (using composite action):

```yaml
- name: Setup Python environment
  uses: ./.github/actions/setup-python-env
```

That's it! The composite action handles:
- ✅ Installing Python 3.11
- ✅ Setting up pip caching
- ✅ Upgrading pip
- ✅ Installing from requirements.txt

---

## Full Updated Workflow Example

Here's what a section of `daily_brands.yml` would look like:

```yaml
jobs:
  run:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1
          sparse-checkout: |
            requirements.txt
            scripts/
            rosters/
            data/processed_serps
            data/processed_articles
            data/daily_counts/
            .github/
          sparse-checkout-cone-mode: false

      - name: Setup Python environment
        uses: ./.github/actions/setup-python-env

      - name: Ensure folders (NEW STRUCTURE)
        run: |
          mkdir -p data/daily_counts
          mkdir -p data/processed_articles
          mkdir -p data/processed_serps
          mkdir -p rosters

      - name: Build brand articles
        env:
          RUN_DATE: ${{ github.event.inputs.date || '' }}
        run: |
          set -euo pipefail
          DATE_TO_RUN="${RUN_DATE:-$(date -u +%F)}"
          echo "Running news_articles_brands.py for ${DATE_TO_RUN}"
          python scripts/news_articles_brands.py --date "${DATE_TO_RUN}"
      
      # ... rest of your workflow steps
```

---

## Apply to All Workflows

You can replace the Python setup steps in ALL these workflows:

- ✅ `daily_brands.yml`
- ✅ `daily_ceos.yml`
- ✅ `backfill_serps.yml`
- ✅ `send_alerts.yml`
- ✅ `aggregate_negative_articles.yml`
- ✅ `fetch-stock-data.yml`
- ✅ `fetch-trends.yml`

**Benefits:**
- 🚀 ~80% faster runs after first execution (5-10s vs 30-45s)
- 🎯 Consistent Python setup across all workflows
- 🔧 Easy to maintain - change once, updates everywhere
- 📦 All dependencies in one place (requirements.txt)
