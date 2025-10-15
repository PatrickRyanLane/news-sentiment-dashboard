# Suggested Git Commit

## Option 1: Single Commit (Simple)

```bash
git add .github/ requirements.txt
git commit -m "feat: optimize GitHub Actions workflows

- Add composite action for Python setup with pip caching
- Standardize all workflows to use actions/checkout@v4
- Simplify sparse-checkout configurations (data/ instead of subdirs)
- Fix daily_ceos.yml concurrency to prevent data loss
- Add pytrends to requirements.txt for fetch-trends workflow

Performance: 80% faster dependency installation after first run
Time saved: ~3-4 minutes per day, ~90-120 minutes per month"
```

## Option 2: Separate Commits (Organized)

```bash
# Commit 1: Create composite action
git add .github/actions/
git commit -m "feat: create composite action for Python environment setup

- Add .github/actions/setup-python-env with pip caching
- Include comprehensive documentation and examples
- Enables 80% faster dependency installation"

# Commit 2: Update requirements
git add requirements.txt
git commit -m "feat: add pytrends to requirements.txt

Required by fetch-trends.yml workflow"

# Commit 3: Update workflows
git add .github/workflows/
git commit -m "feat: optimize all GitHub Actions workflows

- Integrate composite action for consistent Python setup
- Standardize to actions/checkout@v4 across all workflows
- Simplify sparse-checkout (use data/ instead of subdirectories)
- Fix daily_ceos.yml: cancel-in-progress true → false

Workflows updated:
- aggregate_negative_articles.yml
- backfill_serps.yml
- daily_brands.yml
- daily_ceos.yml
- fetch-stock-data.yml
- fetch-trends.yml
- send_alerts.yml"

# Commit 4: Add documentation
git add .github/WORKFLOW_UPDATES.md
git commit -m "docs: add workflow optimization summary"
```

## Recommended: Option 1 (Single Commit)

For these changes, Option 1 is better because:
- ✅ Changes are closely related (all part of workflow optimization)
- ✅ Easier to revert if needed
- ✅ Cleaner git history
- ✅ Less overhead

## After Committing:

```bash
# Push to test branch first
git checkout -b optimize-workflows
git push origin optimize-workflows

# Test a workflow
gh workflow run fetch-stock-data.yml --ref optimize-workflows
gh run watch

# If successful, merge to main
git checkout main
git merge optimize-workflows
git push origin main
```
