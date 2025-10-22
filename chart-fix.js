// Enhanced getDateSeries function with zero-filling for missing dates
function getDateSeries(){
  // First, get ALL dates from the data (not filtered by company)
  const allDatesSet = new Set();
  allCountsRows.forEach(r => {
    if (isISODate(r.date)) allDatesSet.add(r.date);
  });
  serpsDaily.forEach(r => {
    if (isISODate(r.date)) allDatesSet.add(r.date);
  });
  
  // Sort all dates
  const allDates = [...allDatesSet].sort();
  
  // If we have a selected company, we'll fill missing dates with zeros
  if (selectedCompany) {
    console.log(`📊 Building chart data for: ${selectedCompany}`);
    
    // Create maps for this company's actual data
    const companyNewsData = new Map();
    const companySerpData = new Map();
    
    // Collect actual data points for this company
    allCountsRows.forEach(r => {
      if (r.company === selectedCompany) {
        companyNewsData.set(r.date, {
          pos: +r.pos || 0,
          neu: +r.neu || 0,
          neg: +r.neg || 0
        });
      }
    });
    
    serpsDaily.forEach(r => {
      if (r.company === selectedCompany) {
        companySerpData.set(r.date, {
          total: +r.total || 0,
          neg: +r.neg_serp || 0,
          ctrl: +r.ctrl || 0
        });
      }
    });
    
    console.log(`  - Found ${companyNewsData.size} days with article data`);
    console.log(`  - Found ${companySerpData.size} days with SERP data`);
    console.log(`  - Total date range has ${allDates.length} days`);
    
    // Build arrays with zeros for missing dates
    const newsPos = [], newsNeu = [], newsNeg = [], serpNegPct = [], serpCtrlPct = [];
    
    allDates.forEach(date => {
      // News data - use actual data if exists, otherwise zeros
      const newsData = companyNewsData.get(date);
      if (newsData) {
        const total = newsData.pos + newsData.neu + newsData.neg;
        if (total > 0) {
          newsPos.push((newsData.pos / total) * 100);
          newsNeu.push((newsData.neu / total) * 100);
          newsNeg.push((newsData.neg / total) * 100);
        } else {
          newsPos.push(0);
          newsNeu.push(0);
          newsNeg.push(0);
        }
      } else {
        // No data for this date - use null or 0
        // Using null will create gaps in the chart
        // Using 0 will show as zero values
        newsPos.push(null);  // or use 0 for continuous line
        newsNeu.push(null);  // or use 0
        newsNeg.push(null);  // or use 0
      }
      
      // SERP data - use actual data if exists, otherwise zeros
      const serpData = companySerpData.get(date);
      if (serpData && serpData.total > 0) {
        serpNegPct.push((serpData.neg / serpData.total) * 100);
        serpCtrlPct.push((serpData.ctrl / serpData.total) * 100);
      } else {
        // No SERP data for this date
        serpNegPct.push(null);  // or use 0
        serpCtrlPct.push(null);  // or use 0
      }
    });
    
    return {
      dates: allDates,
      newsPos,
      newsNeu,
      newsNeg,
      serpNegPct,
      serpCtrlPct
    };
    
  } else {
    // No company selected - show aggregate data (existing logic)
    const byDate = new Map();
    allCountsRows.forEach(r => {
      const key = r.date;
      if (!byDate.has(key)) byDate.set(key, {pos:0, neu:0, neg:0});
      const b = byDate.get(key);
      b.pos += +r.pos || 0;
      b.neu += +r.neu || 0; 
      b.neg += +r.neg || 0;
    });

    const serpByDate = new Map();
    serpsDaily.forEach(r => {
      const key = r.date;
      if (!serpByDate.has(key)) serpByDate.set(key, {total:0, neg:0, ctrl:0});
      const b = serpByDate.get(key);
      b.total += +r.total || 0;
      b.neg += +r.neg_serp || 0;
      b.ctrl += +r.ctrl || 0;
    });

    const dates = [...new Set([...byDate.keys(), ...serpByDate.keys()])].filter(isISODate).sort();
    const newsPos = [], newsNeu = [], newsNeg = [], serpNegPct = [], serpCtrlPct = [];
    
    dates.forEach(d => {
      const n = byDate.get(d) || {pos:0, neu:0, neg:0};
      const t = n.pos + n.neu + n.neg || 0;
      newsPos.push(t ? n.pos/t*100 : 0);
      newsNeu.push(t ? n.neu/t*100 : 0);
      newsNeg.push(t ? n.neg/t*100 : 0);

      const s = serpByDate.get(d) || {total:0, neg:0, ctrl:0};
      serpNegPct.push(s.total ? s.neg/s.total*100 : 0);
      serpCtrlPct.push(s.total ? s.ctrl/s.total*100 : 0);
    });
    
    return {dates, newsPos, newsNeu, newsNeg, serpNegPct, serpCtrlPct};
  }
}