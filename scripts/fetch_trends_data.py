"""
Fetch Google Trends search volume data for companies in the roster.
Retrieves relative search interest over the past 30 days to align with stock data.
"""

from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import random

# Set up paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROSTER_PATH = PROJECT_ROOT / 'rosters' / 'main-roster.csv'
STOCK_DATA_DIR = PROJECT_ROOT / 'data' / 'stock_prices'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'trends_data'

def fetch_trends_data():
    """
    Fetch Google Trends data for all companies.
    Aligns with the most recent stock data file to ensure date consistency.
    """
    print("Loading roster...")
    roster_df = pd.read_csv(ROSTER_PATH)
    
    # Load the most recent stock data to get the exact dates we need
    stock_files = sorted(STOCK_DATA_DIR.glob('*-stock-data.csv'))
    if not stock_files:
        print("❌ No stock data files found. Please run fetch_stock_data.py first.")
        return None
    
    latest_stock_file = stock_files[-1]
    print(f"Using dates from: {latest_stock_file.name}")
    stock_df = pd.read_csv(latest_stock_file)
    
    # Get unique companies from stock data (those that have stock tickers)
    companies = stock_df['company'].unique()
    print(f"Found {len(companies)} companies to fetch trends for")
    
    # Initialize pytrends
    pytrends = TrendReq(hl='en-US', tz=360)
    
    results = []
    failed_companies = []
    
    for company in companies:
        try:
            print(f"Fetching trends for {company}...")
            
            # Get the date range from the stock data
            stock_row = stock_df[stock_df['company'] == company].iloc[0]
            date_history = stock_row['date_history'].split('|')
            
            # Use the date range from stock data
            start_date = date_history[0]
            end_date = date_history[-1]
            
            # Build the timeframe string for pytrends (YYYY-MM-DD YYYY-MM-DD)
            timeframe = f"{start_date} {end_date}"
            
            # Try company name as search term
            # You might want to customize this - some companies are better searched by ticker
            search_terms = [company]
            
            pytrends.build_payload(search_terms, timeframe=timeframe, geo='US')
            
            # Get interest over time
            interest_df = pytrends.interest_over_time()
            
            if interest_df.empty or company not in interest_df.columns:
                print(f"  ⚠️  No trends data available for {company}")
                failed_companies.append({'company': company, 'reason': 'No data'})
                
                # Add empty result to maintain consistency
                results.append({
                    'company': company,
                    'trends_history': '',
                    'date_history': stock_row['date_history'],
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                continue
            
            # Get the search interest values
            interest_values = interest_df[company].tolist()
            
            # Align trends data with stock dates
            # Trends might return weekly data, so we need to interpolate/align
            trends_dates = interest_df.index.strftime('%Y-%m-%d').tolist()
            
            # Create a mapping of dates to interest values
            trends_map = dict(zip(trends_dates, interest_values))
            
            # Align with stock dates (fill missing dates with interpolation or previous value)
            aligned_trends = []
            for stock_date in date_history:
                if stock_date in trends_map:
                    aligned_trends.append(trends_map[stock_date])
                else:
                    # Use the most recent available value (forward fill)
                    # Find the closest earlier date
                    earlier_values = [v for d, v in trends_map.items() if d <= stock_date]
                    if earlier_values:
                        aligned_trends.append(earlier_values[-1])
                    else:
                        # If no earlier date, use the first available value
                        aligned_trends.append(list(trends_map.values())[0] if trends_map else 0)
            
            # Calculate average search interest
            avg_interest = sum(aligned_trends) / len(aligned_trends) if aligned_trends else 0
            
            results.append({
                'company': company,
                'trends_history': '|'.join(map(str, [int(v) for v in aligned_trends])),
                'date_history': stock_row['date_history'],
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'avg_interest': round(avg_interest, 1)
            })
            
            print(f"  ✓ {company}: Avg interest: {avg_interest:.1f}/100 - {len(aligned_trends)} days")
            
            # Add delay to avoid rate limiting (Google Trends has strict limits)
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            print(f"  ✗ Error fetching trends for {company}: {str(e)}")
            failed_companies.append({'company': company, 'reason': str(e)})
            
            # Add empty result to maintain consistency
            stock_row = stock_df[stock_df['company'] == company].iloc[0]
            results.append({
                'company': company,
                'trends_history': '',
                'date_history': stock_row['date_history'],
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'avg_interest': 0
            })
            
            # Add delay even on failure
            time.sleep(random.uniform(1, 2))
    
    # Create DataFrame
    trends_df = pd.DataFrame(results)
    
    # Save results with date-stamped filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    output_file = OUTPUT_DIR / f'{today}-trends-data.csv'
    
    trends_df.to_csv(output_file, index=False)
    print(f"\n✓ Trends data saved to {output_file}")
    print(f"  Successfully fetched: {len([r for r in results if r['trends_history']])} companies")
    print(f"  Failed/Empty: {len(failed_companies)} companies")
    
    # Save failed companies for debugging
    if failed_companies:
        failed_df = pd.DataFrame(failed_companies)
        failed_file = OUTPUT_DIR / f'{today}-failed-trends.csv'
        failed_df.to_csv(failed_file, index=False)
        print(f"  Failed companies saved to {failed_file}")
    
    return trends_df

if __name__ == "__main__":
    print("=" * 60)
    print("Google Trends Data Fetcher")
    print("=" * 60)
    print("\n⚠️  NOTE: Google Trends has rate limits. This script includes")
    print("delays between requests. For 100 companies, expect ~3-5 minutes.\n")
    
    fetch_trends_data()
