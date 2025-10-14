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

# Suppress the pandas FutureWarning from pytrends
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

# Set up paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ROSTER_PATH = PROJECT_ROOT / 'rosters' / 'main-roster.csv'
STOCK_DATA_DIR = PROJECT_ROOT / 'data' / 'stock_prices'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'trends_data'

# Rate limit settings
MIN_DELAY = 2.0  # Minimum delay between requests (increased from 1)
MAX_DELAY = 4.0  # Maximum delay between requests (increased from 2)
RETRY_DELAY = 10.0  # Delay after rate limit error
MAX_RETRIES = 2  # Number of retries for rate-limited requests

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
    consecutive_rate_limits = 0
    MAX_CONSECUTIVE_RATE_LIMITS = 5  # Stop if we hit 5 rate limits in a row
    
    for idx, company in enumerate(companies):
        # Early exit if we're clearly rate limited
        if consecutive_rate_limits >= MAX_CONSECUTIVE_RATE_LIMITS:
            print(f"\n⚠️  Hit {MAX_CONSECUTIVE_RATE_LIMITS} consecutive rate limits. Stopping to avoid further blocking.")
            print(f"   Collected data for {len([r for r in results if r['trends_history']])} companies.")
            print(f"   Will resume from {company} on next run.\n")
            
            # Add empty results for remaining companies
            for remaining_company in companies[idx:]:
                stock_row = stock_df[stock_df['company'] == remaining_company].iloc[0]
                results.append({
                    'company': remaining_company,
                    'trends_history': '',
                    'date_history': stock_row['date_history'],
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'avg_interest': 0
                })
            break
        
        retry_count = 0
        success = False
        
        while retry_count <= MAX_RETRIES and not success:
            try:
                if retry_count > 0:
                    print(f"  🔄 Retry {retry_count}/{MAX_RETRIES} for {company}...")
                else:
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
                
                # Re-initialize pytrends for each request to avoid session issues
                # Note: Removed retries/backoff_factor due to urllib3 compatibility issues
                # We handle retries at the script level instead
                pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
                
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
                    success = True
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
                
                success = True
                consecutive_rate_limits = 0  # Reset counter on success
                
                # Add delay to avoid rate limiting - longer delay for successful requests
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                time.sleep(delay)
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error (429)
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    retry_count += 1
                    consecutive_rate_limits += 1
                    
                    if retry_count <= MAX_RETRIES:
                        # Exponential backoff: wait longer with each retry
                        backoff_delay = RETRY_DELAY * (2 ** (retry_count - 1))
                        print(f"  ⚠️  Rate limited (429). Waiting {backoff_delay:.0f}s before retry...")
                        time.sleep(backoff_delay)
                    else:
                        print(f"  ✗ Rate limit exceeded for {company} after {MAX_RETRIES} retries")
                        print(f"     (Consecutive rate limits: {consecutive_rate_limits})")
                        failed_companies.append({'company': company, 'reason': 'Rate limit (429)'})
                        
                        # Add empty result
                        stock_row = stock_df[stock_df['company'] == company].iloc[0]
                        results.append({
                            'company': company,
                            'trends_history': '',
                            'date_history': stock_row['date_history'],
                            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'avg_interest': 0
                        })
                        
                        # Longer delay after giving up on rate limit
                        time.sleep(RETRY_DELAY)
                else:
                    # Non-rate-limit error - don't retry
                    print(f"  ✗ Error fetching trends for {company}: {error_msg}")
                    failed_companies.append({'company': company, 'reason': error_msg})
                    
                    # Add empty result
                    stock_row = stock_df[stock_df['company'] == company].iloc[0]
                    results.append({
                        'company': company,
                        'trends_history': '',
                        'date_history': stock_row['date_history'],
                        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'avg_interest': 0
                    })
                    
                    success = True  # Don't retry non-rate-limit errors
                    consecutive_rate_limits = 0  # Reset counter for non-rate-limit errors
                    
                    # Normal delay
                    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    
    # Create DataFrame
    trends_df = pd.DataFrame(results)
    
    # Save results with date-stamped filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    output_file = OUTPUT_DIR / f'{today}-trends-data.csv'
    
    trends_df.to_csv(output_file, index=False)
    
    # Calculate statistics
    total_companies = len(results)
    successful = len([r for r in results if r['trends_history']])
    rate_limited = len([f for f in failed_companies if '429' in str(f.get('reason', ''))])
    other_failures = len(failed_companies) - rate_limited
    
    print(f"\n✓ Trends data saved to {output_file}")
    print(f"  Total companies: {total_companies}")
    print(f"  Successfully fetched: {successful}")
    print(f"  Rate limited (429): {rate_limited}")
    print(f"  Other failures: {other_failures}")
    
    if rate_limited > 0:
        print(f"\n💡 TIP: {rate_limited} companies hit rate limits.")
        print(f"   This is normal. Next scheduled run will try again.")
        print(f"   Consider:")
        print(f"   • Running during off-peak hours (evening/night)")
        print(f"   • Increasing delays in the script")
        print(f"   • Running less frequently (once daily max)")
    
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
