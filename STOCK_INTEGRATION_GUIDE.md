# Stock Price Integration Guide

This guide explains how to integrate the stock price data into your dashboards.

## Overview

The stock data is now fetched daily and stored in `data/stock_prices/stock_data_YYYY-MM-DD.csv` with the following columns:
- `ticker`: Stock ticker symbol
- `company`: Company name
- `opening_price`: Opening price for the day
- `seven_day_change_pct`: 7-day percentage change
- `price_history`: Pipe-separated list of closing prices (last 7 days)
- `date_history`: Pipe-separated list of dates
- `last_updated`: Timestamp

## Dashboard Integration Steps

### 1. Load Stock Data in JavaScript

Add this function to load the latest stock data:

```javascript
async function loadStockData() {
    try {
        // Get today's date
        const today = new Date().toISOString().split('T')[0];
        
        // Try to load today's file, fall back to yesterday if not available
        let stockData = {};
        
        try {
            const response = await fetch(`data/stock_prices/stock_data_${today}.csv`);
            const text = await response.text();
            stockData = parseStockCSV(text);
        } catch {
            // Try yesterday's file
            const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
            const response = await fetch(`data/stock_prices/stock_data_${yesterday}.csv`);
            const text = await response.text();
            stockData = parseStockCSV(text);
        }
        
        return stockData;
    } catch (error) {
        console.error('Error loading stock data:', error);
        return {};
    }
}

function parseStockCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');
    const stockData = {};
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const company = values[1];
        
        stockData[company] = {
            ticker: values[0],
            company: values[1],
            openingPrice: parseFloat(values[2]),
            sevenDayChange: parseFloat(values[3]),
            priceHistory: values[4] ? values[4].split('|').map(p => parseFloat(p)) : [],
            dateHistory: values[5] ? values[5].split('|') : [],
            lastUpdated: values[6]
        };
    }
    
    return stockData;
}
```

### 2. Update Table to Display Stock Data

Modify your table rendering to include stock columns:

```javascript
async function renderTable() {
    const stockData = await loadStockData();
    
    // ... existing code to build table rows ...
    
    // Add stock columns to header
    const headerRow = `
        <th>Company</th>
        <th>Stock Price</th>
        <th>7-Day Change</th>
        <th>Sentiment</th>
        <!-- other columns -->
    `;
    
    // Add stock data to each row
    rows.forEach(row => {
        const stock = stockData[row.company];
        
        if (stock) {
            const changeClass = stock.sevenDayChange >= 0 ? 'positive' : 'negative';
            const changeSymbol = stock.sevenDayChange >= 0 ? '▲' : '▼';
            
            row.stockPrice = `$${stock.openingPrice.toFixed(2)}`;
            row.stockChange = `
                <span class="${changeClass}">
                    ${changeSymbol} ${Math.abs(stock.sevenDayChange).toFixed(2)}%
                </span>
            `;
        } else {
            row.stockPrice = 'N/A';
            row.stockChange = 'N/A';
        }
    });
}
```

### 3. Add CSS for Stock Data Styling

```css
/* Stock price styling */
.positive {
    color: #22c55e;
    font-weight: 600;
}

.negative {
    color: #ef4444;
    font-weight: 600;
}

.stock-price {
    font-family: 'Courier New', monospace;
    font-weight: 500;
}

/* Stock chart modal */
#stockChartModal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0,0,0,0.5);
}

#stockChartContent {
    background-color: white;
    margin: 5% auto;
    padding: 20px;
    width: 80%;
    max-width: 800px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#stockChart {
    width: 100%;
    height: 400px;
}
```

### 4. Add Stock Chart Modal with Chart.js

First, add Chart.js to your HTML head:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

Then add the modal HTML before closing body tag:

```html
<!-- Stock Chart Modal -->
<div id="stockChartModal">
    <div id="stockChartContent">
        <span class="close" onclick="closeStockChart()">&times;</span>
        <h2 id="chartCompanyName"></h2>
        <canvas id="stockChart"></canvas>
    </div>
</div>
```

Add JavaScript functions for chart:

```javascript
let stockChart = null;

function showStockChart(company) {
    const stock = globalStockData[company];
    if (!stock || !stock.priceHistory.length) {
        alert('No stock data available for this company');
        return;
    }
    
    // Show modal
    document.getElementById('stockChartModal').style.display = 'block';
    document.getElementById('chartCompanyName').textContent = 
        `${company} (${stock.ticker}) - 7-Day Price History`;
    
    // Destroy existing chart if it exists
    if (stockChart) {
        stockChart.destroy();
    }
    
    // Create new chart
    const ctx = document.getElementById('stockChart').getContext('2d');
    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: stock.dateHistory,
            datasets: [{
                label: 'Closing Price ($)',
                data: stock.priceHistory,
                borderColor: stock.sevenDayChange >= 0 ? '#22c55e' : '#ef4444',
                backgroundColor: stock.sevenDayChange >= 0 ? 
                    'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                borderWidth: 2,
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toFixed(2);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

function closeStockChart() {
    document.getElementById('stockChartModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('stockChartModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}
```

### 5. Make Rows Clickable to Show Chart

Update your table rows to be clickable:

```javascript
// Add click handler to each row
row.addEventListener('click', () => {
    showStockChart(rowData.company);
});

// Add visual feedback
row.style.cursor = 'pointer';
row.addEventListener('mouseenter', () => {
    row.style.backgroundColor = '#f3f4f6';
});
row.addEventListener('mouseleave', () => {
    row.style.backgroundColor = '';
});
```

## Example: Complete Integration

Here's a minimal complete example:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard with Stock Data</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .positive { color: #22c55e; font-weight: 600; }
        .negative { color: #ef4444; font-weight: 600; }
        tr { cursor: pointer; }
        tr:hover { background-color: #f3f4f6; }
    </style>
</head>
<body>
    <table id="dataTable">
        <thead>
            <tr>
                <th>Company</th>
                <th>Stock Price</th>
                <th>7-Day Change</th>
            </tr>
        </thead>
        <tbody id="tableBody"></tbody>
    </table>
    
    <!-- Stock Chart Modal -->
    <div id="stockChartModal" style="display:none;">
        <div id="stockChartContent">
            <span onclick="closeStockChart()" style="cursor:pointer;">&times;</span>
            <h2 id="chartCompanyName"></h2>
            <canvas id="stockChart"></canvas>
        </div>
    </div>
    
    <script>
        // Include all the JavaScript functions from above
        // loadStockData(), parseStockCSV(), renderTable(), 
        // showStockChart(), closeStockChart()
    </script>
</body>
</html>
```

## Testing

1. **Manual Test**: Run the script locally:
   ```bash
   python scripts/fetch_stock_data.py
   ```

2. **Check Output**: Verify the CSV file in `data/stock_prices/`

3. **Test Dashboard**: Open your HTML dashboard and verify:
   - Stock prices display correctly
   - 7-day changes show with correct colors
   - Clicking a company opens the chart modal
   - Chart displays historical data

## Troubleshooting

- **"No data available"**: The stock data file may not exist yet. Run the workflow manually or wait for the scheduled run.
- **Chart not displaying**: Ensure Chart.js is loaded before your script runs.
- **Ticker symbol issues**: Some tickers in your roster may not be valid or may have changed. Check the `failed_tickers_YYYY-MM-DD.csv` file.

## Next Steps

After integrating stock data:
1. Update all dashboard HTML files (brand-dashboard.html, ceo-dashboard.html, sectors.html)
2. Test on GitHub Pages
3. Consider adding:
   - Volume data
   - Market cap
   - Additional chart types (candlestick, etc.)
   - Comparison charts for multiple companies
