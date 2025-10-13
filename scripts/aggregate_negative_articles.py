#!/usr/bin/env python3
"""
DEBUG VERSION: Aggregate negative articles with detailed logging
This will help us see what's happening with brand vs CEO articles
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone


def process_articles(file_path, article_type):
    """Process articles with detailed logging"""
    if not file_path.exists():
        print(f"  ⚠️  File not found: {file_path.name}")
        return []
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            print(f"  ⚠️  File is empty: {file_path.name}")
            return []
        
        print(f"\n  📄 Processing {file_path.name}")
        print(f"     Total rows: {len(df)}")
        
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        print(f"     Columns: {list(df.columns)}")
        
        # Ensure required columns exist
        required_cols = ['ceo', 'company', 'sentiment', 'title']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  ❌ Missing columns: {missing}")
            return []
        
        # Clean up data
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
        df['ceo'] = df['ceo'].astype(str).str.strip()
        df['company'] = df['company'].astype(str).str.strip()
        df['title'] = df['title'].astype(str).str.strip()
        
        # Show sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        print(f"     Sentiment distribution:")
        for sent, count in sentiment_counts.items():
            print(f"       - {sent}: {count}")
        
        # Filter for negative sentiment only
        negative = df[df['sentiment'] == 'negative']
        print(f"     Negative articles: {len(negative)}")
        
        if negative.empty:
            print(f"  ℹ️  No negative articles found")
            return []
        
        summary_data = []
        
        # Group by company/CEO and aggregate
        for (ceo, company), group in negative.groupby(['ceo', 'company']):
            if not ceo or not company or ceo == 'nan' or company == 'nan':
                continue
            
            count = len(group)
            print(f"     ✓ {company} ({ceo}): {count} negative articles [type={article_type}]")
            
            # Get top 3 headlines
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
        
        print(f"     Summary rows created: {len(summary_data)}")
        return summary_data
    
    except Exception as e:
        print(f"  ❌ Error processing {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_negative_summary(days_back=7):  # Default to 7 days for debugging
    """Create aggregated summary with detailed logging"""
    articles_dir = Path("data/processed_articles")
    output_file = Path("data/negative-articles-summary.csv")
    
    if not articles_dir.exists():
        print(f"❌ Articles directory not found: {articles_dir}")
        return
    
    all_summary_data = []
    today = datetime.now(timezone.utc)
    
    print(f"\n{'='*60}")
    print(f"🔍 DEBUG MODE: Scanning last {days_back} days")
    print(f"{'='*60}")
    
    days_processed = 0
    ceo_files_found = 0
    brand_files_found = 0
    
    for i in range(days_back):
        date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"\n📅 Date: {date}")
        
        # Process CEO articles
        ceo_file = articles_dir / f"{date}-ceo-articles-modal.csv"
        if ceo_file.exists():
            ceo_files_found += 1
            print(f"  ✓ Found CEO file")
            ceo_data = process_articles(ceo_file, 'ceo')
            for item in ceo_data:
                item['date'] = date
                all_summary_data.append(item)
        else:
            print(f"  ⚠️  No CEO file")
        
        # Process brand articles
        brand_file = articles_dir / f"{date}-brand-articles-modal.csv"
        if brand_file.exists():
            brand_files_found += 1
            print(f"  ✓ Found brand file")
            brand_data = process_articles(brand_file, 'brand')
            for item in brand_data:
                item['date'] = date
                all_summary_data.append(item)
        else:
            print(f"  ⚠️  No brand file")
        
        if ceo_file.exists() or brand_file.exists():
            days_processed += 1
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Days processed: {days_processed}")
    print(f"CEO files found: {ceo_files_found}")
    print(f"Brand files found: {brand_files_found}")
    print(f"Total summary rows: {len(all_summary_data)}")
    
    # Create summary DataFrame
    if all_summary_data:
        summary_df = pd.DataFrame(all_summary_data)
        summary_df = summary_df.sort_values(['company', 'date', 'article_type'])
        summary_df = summary_df[['date', 'company', 'ceo', 'negative_count', 'top_headlines', 'article_type']]
        
        # Show breakdown by type
        type_counts = summary_df.groupby('article_type').size()
        print(f"\nRows by article type:")
        for atype, count in type_counts.items():
            print(f"  - {atype}: {count}")
        
        # Show sample companies with both types
        print(f"\n{'='*60}")
        print(f"🔍 COMPANIES WITH BOTH CEO AND BRAND ARTICLES")
        print(f"{'='*60}")
        
        companies_with_both = []
        for company in summary_df['company'].unique():
            company_data = summary_df[summary_df['company'] == company]
            types = company_data['article_type'].unique()
            if len(types) > 1:
                companies_with_both.append(company)
                print(f"\n✓ {company}")
                for atype in ['ceo', 'brand']:
                    type_data = company_data[company_data['article_type'] == atype]
                    if not type_data.empty:
                        total = type_data['negative_count'].sum()
                        dates = len(type_data)
                        print(f"  - {atype}: {total} articles across {dates} dates")
        
        if not companies_with_both:
            print("\n⚠️  WARNING: No companies found with BOTH CEO and brand articles!")
            print("This might explain why you only see CEO counts in the tooltip.")
            print("\nCompanies with CEO articles only:")
            ceo_only = summary_df[summary_df['article_type'] == 'ceo']['company'].unique()
            for co in ceo_only[:5]:
                print(f"  - {co}")
            print("\nCompanies with brand articles only:")
            brand_only = summary_df[summary_df['article_type'] == 'brand']['company'].unique()
            for co in brand_only[:5]:
                print(f"  - {co}")
    else:
        print("\n⚠️  No negative articles found")
        summary_df = pd.DataFrame(columns=[
            'date', 'company', 'ceo', 'negative_count', 'top_headlines', 'article_type'
        ])
    
    # Write to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Created {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='DEBUG: Aggregate negative articles with detailed logging'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='Number of days to look back (default: 7 for debugging)'
    )
    
    args = parser.parse_args()
    create_negative_summary(days_back=args.days_back)
    return 0


if __name__ == "__main__":
    exit(main())
