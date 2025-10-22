// Add these options to both newsChart and serpChart configurations:

const commonOpts = {
  responsive: true, 
  maintainAspectRatio: false,
  layout: { padding: { bottom: 24 } },
  scales: {
    x: { 
      ticks: { color: '#ebf2f2' }, 
      grid: { color: 'rgba(255,255,255,.05)' } 
    },
    y: { 
      ticks: { color: '#ebf2f2', stepSize: 20, callback: v => pct(v) },
      grid: { color: 'rgba(255,255,255,.06)' }, 
      suggestedMin: 0, 
      suggestedMax: 100 
    }
  },
  plugins: {
    legend: { 
      labels: { color: '#ebf2f2' } 
    },
    title: { 
      display: true, 
      text: who, 
      color: '#ebf2f2', 
      font: { weight: 'bold', size: 14 },
      padding: { top: 10, bottom: 6 } 
    },
    tooltip: { 
      callbacks: {
        label: (ctx) => `${ctx.dataset?.label ? ctx.dataset.label + ': ' : ''}${pct(typeof ctx.parsed.y === 'number' ? ctx.parsed.y : ctx.parsed)}`
      }
    }
  },
  // ADD THESE OPTIONS FOR HANDLING SPARSE DATA:
  spanGaps: false,  // Don't connect points across null values
  interaction: {
    mode: 'index',
    intersect: false
  }
};

// For the line chart (serpChart), also add:
serpChart = new Chart(sh, {
  type: 'line',
  data: {
    labels: d,
    datasets: [
      {
        label: 'Daily Negative SERP %',
        data: sN,
        tension: 0.2,
        borderWidth: SOLID_WIDTH,
        fill: false,
        borderColor: NEG_COLOR,
        pointBackgroundColor: NEG_COLOR,
        pointBorderColor: NEG_COLOR,
        spanGaps: false  // ADD THIS to each dataset
      },
      // ... other datasets
    ]
  },
  options: { ...commonOpts }
});