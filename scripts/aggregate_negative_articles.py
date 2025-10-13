#!/usr/bin/env python3
"""
Aggregate negative article data for stock chart heatmap visualization.

Reads both CEO and brand articles for the last 90 days and creates a 
unified summary for fast frontend loading.

Output: data/negative-articles-summary.csv
Columns: date, company, ceo, negative_count, top_headlines, article_type

Usage:
    python scripts/aggregate_negative_articles.py [--days-back 90]
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone


def process_articles(file_path, article_type):
    """
    Process a single articles file and return negative article summary.
    
    Args:
        file_path: Path to the *-articles-modal.csv file
        article_type: Either 'ceo' or 'brand'
    
    Returns:
        List of dicts with negative article summaries
    """
    if not file_path.exists():
        return []
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            return []
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure required columns exist
        required_cols = ['ceo', 'company', 'sentiment', 'title']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️  Missing column '{col}' in {file_path.name}")
                return []
        
        # Clean up data
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
        df['ceo'] = df['ceo'].astype(str).str.strip()
        df['company'] = df['company'].astype(str).str.strip()
        df['title'] = df['title'].astype(str).str.strip()
        
        # Filter for negative sentiment only
        negative = df[df['sentiment'] == 'negative']
        
        if negative.empty:
            return []
        
        summary_data = []
        
        # Group by company/CEO and aggregate
        for (ceo, company), group in negative.groupby(['ceo', 'company']):
            if not ceo or not company or ceo == 'nan' or company == 'nan':
                continue
                
            count = len(group)
            
            # Get top 3 headlines (truncated to 80 chars each)
            headlines = []
            for title in group['title'].head(3):
                title_str = str(title).strip()
                if len(title_str) > 80:
                    title_str = title_str[:77] + '...'
                headlines.append(title_str)
            
            summary_data.append({
                'ceo': ceo,
                'company': company,
                'negative_count': count,
                'top_headlines': '|'.join(headlines),
                'article_type': article_type
            })
        
        return summary_data
    
    except Exception as e:
        print(f"⚠️  Error processing {file_path.name}: {e}")
        return []


def create_negative_summary(days_back=90):
    """
    Create aggregated negative articles summary from last N days.
    
    Args:
        days_back: Number of days to look back (default 90)
    """
    articles_dir = Path("data/processed_articles")
    output_file = Path("data/negative-articles-summary.csv")
    
    if not articles_dir.exists():
        print(f"❌ Articles directory not found: {articles_dir}")
        return
    
    all_summary_data = []
    today = datetime.now(timezone.utc)
    
    print(f"🔍 Scanning last {days_back} days for negative articles...")
    
    days_processed = 0
    ceo_files_found = 0
    brand_files_found = 0
    
    for i in range(days_back):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        
        # Process CEO articles
        ceo_file = articles_dir / f"{date}-ceo-articles-modal.csv"
        if ceo_file.exists():
            ceo_files_found += 1
            ceo_data = process_articles(ceo_file, 'ceo')
            for item in ceo_data:
                item['date'] = date
                all_summary_data.append(item)
        
        # Process brand articles
        brand_file = articles_dir / f"{date}-brand-articles-modal.csv"
        if brand_file.exists():
            brand_files_found += 1
            brand_data = process_articles(brand_file, 'brand')
            for item in brand_data:
                item['date'] = date
                all_summary_data.append(item)
        
        if ceo_file.exists() or brand_file.exists():
            days_processed += 1
    
    print(f"📁 Files found: {ceo_files_found} CEO, {brand_files_found} brand ({days_processed} days with data)")
    
    # Create summary DataFrame
    if all_summary_data:
        summary_df = pd.DataFrame(all_summary_data)
        summary_df = summary_df.sort_values(['company', 'date', 'article_type'])
        summary_df = summary_df[['date', 'company', 'ceo', 'negative_count', 'top_headlines', 'article_type']]
    else:
        print("⚠️  No negative articles found in the specified time range")
        # Create empty file with headers
        summary_df = pd.DataFrame(columns=[
            'date', 'company', 'ceo', 'negative_count', 'top_headlines', 'article_type'
        ])
    
    # Write to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Created {output_file}")
    print(f"📊 Total rows: {len(summary_df):,}")
    
    if not summary_df.empty:
        file_size_kb = output_file.stat().st_size / 1024
        print(f"📊 File size: {file_size_kb:.1f} KB")
        
        ceo_count = len(summary_df[summary_df['article_type'] == 'ceo'])
        brand_count = len(summary_df[summary_df['article_type'] == 'brand'])
        
        print(f"🎯 CEO articles: {ceo_count:,}")
        print(f"🏢 Brand articles: {brand_count:,}")
        print(f"📅 Date range: {summary_df['date'].min()} to {summary_df['date'].max()}")
        
        # Show some stats
        companies = summary_df['company'].nunique()
        avg_per_company = len(summary_df) / companies if companies > 0 else 0
        print(f"🏭 Companies with negative coverage: {companies}")
        print(f"📈 Average negative articles per company: {avg_per_company:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate negative articles for stock chart visualization'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=90,
        help='Number of days to look back (default: 90)'
    )
    
    args = parser.parse_args()
    
    if args.days_back < 1:
        print("❌ --days-back must be at least 1")
        return 1
    
    create_negative_summary(days_back=args.days_back)
    return 0


if __name__ == "__main__":
    exit(main())
