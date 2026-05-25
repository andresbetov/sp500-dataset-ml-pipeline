/**
 * Ticker Analysis Page Logic
 */

let tickerData = null;
let modelInfo = null;

document.addEventListener('DOMContentLoaded', async () => {
    await initTickerPage();
});

/**
 * Initialize ticker page
 */
async function initTickerPage() {
    try {
        const params = getQueryParams();
        const ticker = params.ticker;

        if (!ticker) {
            showToast('Ticker no especificado', 'error');
            window.location.href = 'screener.html';
            return;
        }

        // Load ticker data
        await loadTickerData(ticker);

        // Load model info for feature importance
        modelInfo = await api.modelInfo();
        loadFeatureImportance();

        // Setup event listeners
        document.getElementById('backBtn').addEventListener('click', () => {
            history.back();
        });

        document.getElementById('refreshPred').addEventListener('click', async () => {
            cache.clear(`predict_${ticker}`);
            await loadTickerData(ticker);
            showToast('Predicción actualizada', 'success');
        });

        document.getElementById('exportDataBtn').addEventListener('click', exportData);
        document.getElementById('exportChartBtn').addEventListener('click', exportChart);
        document.getElementById('openScreenerBtn').addEventListener('click', () => {
            window.location.href = 'screener.html';
        });

        // Setup category dropdown
        const catSelect = document.getElementById('indicatorCategory');
        if (catSelect) {
            catSelect.addEventListener('change', (e) => {
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                document.getElementById(e.target.value).classList.add('active');
            });
        }

        // Activate first tab
        document.querySelector('.tab-pane')?.classList.add('active');
    } catch (error) {
        handleAPIError(error);
    }
}

/**
 * Load ticker prediction data
 */
async function loadTickerData(ticker) {
    try {
        const cacheKey = `predict_${ticker}`;
        tickerData = cache.get(cacheKey);

        if (!tickerData) {
            tickerData = await api.predict(ticker, {
                period: settings.period,
                history_limit: 252,
            });
            cache.set(cacheKey, tickerData);
        }

        // Update header
        document.getElementById('tickerTitle').textContent = `${tickerData.ticker}`;

        const latestPrice = tickerData.metadata.data_quality.date_range.max;
        const volatilityDisplay = formatNumber(tickerData.latest.volatility_daily);
        document.getElementById('tickerSubtitle').textContent = `Volatilidad: ${volatilityDisplay} daily | Última actualización: ${formatDate(latestPrice)}`;

        // Update prediction card
        const latestVol = tickerData.latest.volatility_daily;
        document.getElementById('predVolatility').textContent = formatNumber(latestVol);
        document.getElementById('annualized').textContent = formatNumber(tickerData.latest.volatility_annualized) + ' annual';
        
        const vsMean = tickerData.latest.vs_historical_mean_pct;
        const vsMeanElem = document.getElementById('vsMean');
        vsMeanElem.textContent = formatPercentage(vsMean);
        vsMeanElem.className = 'value-trend ' + (vsMean >= 0 ? 'up' : 'down');
        
        document.getElementById('confidence').textContent = '78.21%'; // From model

        // Update data quality
        const dq = tickerData.metadata.data_quality;
        document.getElementById('dqDownloaded').textContent = formatLargeNumber(dq.downloaded_rows);
        document.getElementById('dqValid').textContent = formatLargeNumber(dq.rows_after_preparation);
        document.getElementById('dqComplete').textContent = formatLargeNumber(dq.valid_rows);
        document.getElementById('dqDateRange').textContent =
            `${formatDate(dq.date_range.min)} → ${formatDate(dq.date_range.max)}`;

        // Update context info
        const contextLastUpdate = document.getElementById('contextLastUpdate');
        if (contextLastUpdate) {
            contextLastUpdate.textContent = formatDate(tickerData.metadata.data_quality.date_range.max);
        }

        // Update indicators (would need to be populated from API in real scenario)
        // For now, using placeholder values
        updateIndicators();

        // Update history stats
        updateHistoryStats();

        // Create history chart
        createHistoryChart();
    } catch (error) {
        handleAPIError(error);
    }
}

/**
 * Update indicators display
 */
function updateIndicators() {
    // Placeholder values - in real app these would come from API
    const indicators = {
        ema12: '128.10',
        ema26: '125.45',
        sma50: '120.34',
        priceSma50: '+6.6%',
        zscoceSma: '+1.23σ',
        rsi14: '65.4',
        macd: '+0.0145',
        macdSignal: '0.0089',
        macdHist: '0.0056',
        atr14: '2.34',
        vol10: '0.0156',
        vol20: '0.0189',
        logReturn: '0.00234',
        avgVol20: '42.3M',
        volZscore: '+0.89σ',
        volChange: '+15.2%',
        regimeId: tickerData.metadata.market_regime.id,
        regimeLabel: getRegimeLabel(tickerData.metadata.market_regime.id),
        regimePeriod: '2022-26',
        contextLastUpdate: formatDate(tickerData.latest.date),
    };

    Object.entries(indicators).forEach(([key, value]) => {
        const elem = document.getElementById(key);
        if (elem) {
            elem.textContent = value;
        }
    });

    // Add regime color class
    const regimeLabelElem = document.getElementById('regimeLabel');
    if (regimeLabelElem) {
        regimeLabelElem.className = `regime-${tickerData.metadata.market_regime.id}`;
    }
}

/**
 * Update historical statistics
 */
function updateHistoryStats() {
    const stats = tickerData.history_stats;

    document.getElementById('histMean').textContent = formatNumber(stats.mean);
    document.getElementById('histMedian').textContent = formatNumber(stats.median);
    document.getElementById('histStd').textContent = formatNumber(stats.std);
    
    const histMin = document.getElementById('histMin');
    histMin.textContent = formatNumber(stats.min);
    histMin.className = 'stat-value accent-green';
    
    const histMax = document.getElementById('histMax');
    histMax.textContent = formatNumber(stats.max);
    histMax.className = 'stat-value accent-red';
    
    document.getElementById('histQ1Q3').textContent =
        `${formatNumber(stats.percentiles.p25)} - ${formatNumber(stats.percentiles.p75)}`;
}

/**
 * Create history chart with enhanced visuals
 */
function createHistoryChart() {
    const canvas = document.getElementById('historyChart');
    if (!canvas || !tickerData.history || tickerData.history.length === 0) return;

    const container = canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = 360 * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = '360px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = 360;
    const pad = { top: 20, right: 20, bottom: 45, left: 60 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;

    const history = tickerData.history;
    const values = history.map(h => h.volatility_daily);
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    const range = maxVal - minVal || 1;

    // Compute SMA-5
    const sma5 = values.map((_, i) => {
        const slice = values.slice(Math.max(0, i - 4), i + 1);
        return slice.reduce((a, b) => a + b, 0) / slice.length;
    });

    const meanVal = values.reduce((a, b) => a + b, 0) / values.length;

    // Helpers
    const xPos = i => pad.left + (i / (history.length - 1)) * cw;
    const yPos = v => pad.top + ch - ((v - minVal) / range) * ch;

    // ── Background ──
    ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
    ctx.fillRect(0, 0, W, H);

    // ── Grid lines (horizontal) ──
    const nGrid = 5;
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= nGrid; i++) {
        const y = pad.top + (ch * i) / nGrid;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
    }

    // ── Area fill (gradient) ──
    const fillGrad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
    fillGrad.addColorStop(0, 'rgba(30, 64, 175, 0.25)');
    fillGrad.addColorStop(0.5, 'rgba(30, 64, 175, 0.1)');
    fillGrad.addColorStop(1, 'rgba(30, 64, 175, 0.02)');

    ctx.fillStyle = fillGrad;
    ctx.beginPath();
    ctx.moveTo(xPos(0), H - pad.bottom);
    history.forEach((_, i) => ctx.lineTo(xPos(i), yPos(values[i])));
    ctx.lineTo(xPos(history.length - 1), H - pad.bottom);
    ctx.closePath();
    ctx.fill();

    // ── Mean reference line ──
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    const yMean = yPos(meanVal);
    ctx.moveTo(pad.left, yMean);
    ctx.lineTo(W - pad.right, yMean);
    ctx.stroke();
    ctx.setLineDash([]);

    // Mean label
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Media', pad.left + 4, yMean - 2);

    // ── SMA-5 trend line ──
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    sma5.forEach((v, i) => {
        const method = i === 0 ? 'moveTo' : 'lineTo';
        ctx[method](xPos(i), yPos(v));
    });
    ctx.stroke();

    // ── Main volatility line (colored segments) ──
    for (let i = 1; i < history.length; i++) {
        const ratio = (values[i] - minVal) / range;
        const r = Math.round(30 + ratio * 200);
        const g = Math.round(180 - ratio * 150);
        const b = Math.round(80 - ratio * 60);
        ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(xPos(i - 1), yPos(values[i - 1]));
        ctx.lineTo(xPos(i), yPos(values[i]));
        ctx.stroke();
    }

    // ── Data points ──
    const step = Math.max(1, Math.floor(history.length / 60));
    history.forEach((_, i) => {
        if (i % step !== 0 && i !== history.length - 1) return;
        const x = xPos(i);
        const y = yPos(values[i]);
        const ratio = (values[i] - minVal) / range;

        ctx.fillStyle = `rgb(${Math.round(30 + ratio * 200)}, ${Math.round(180 - ratio * 150)}, ${Math.round(80 - ratio * 60)})`;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
    });

    // ── Last point highlight ──
    const lastX = xPos(history.length - 1);
    const lastY = yPos(values[values.length - 1]);

    ctx.fillStyle = 'rgba(30, 64, 175, 0.12)';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 14, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#1E40AF';
    ctx.beginPath();
    ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(30, 64, 175, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 8, 0, Math.PI * 2);
    ctx.stroke();

    // ── Y-axis labels ──
    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= nGrid; i++) {
        const v = maxVal - (range * i) / nGrid;
        const y = pad.top + (ch * i) / nGrid;
        ctx.fillText((v * 100).toFixed(2) + '%', pad.left - 8, y);
    }

    // Y-axis title
    ctx.save();
    ctx.translate(14, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#64748B';
    ctx.font = '10px sans-serif';
    ctx.fillText('Volatilidad Diaria', 0, 0);
    ctx.restore();

    // ── X-axis labels (monthly) ──
    ctx.fillStyle = '#94A3B8';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const monthLabels = {};
    history.forEach((h, i) => {
        const d = new Date(h.date);
        const key = d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0');
        if (!monthLabels[key]) {
            monthLabels[key] = { idx: i, label: d.toLocaleDateString('es', { month: 'short', year: '2-digit' }) };
        }
    });

    Object.values(monthLabels).forEach(({ idx, label }) => {
        ctx.fillText(label, xPos(idx), H - pad.bottom + 8);
    });

    // ── Legend ──
    const legendY = pad.top + 6;
    const legendItems = [
        { color: '#1E40AF', label: 'Volatilidad' },
        { color: 'rgba(251, 191, 36, 0.7)', label: 'SMA-5' },
        { color: 'rgba(148, 163, 184, 0.6)', label: 'Media' },
    ];

    let legendX = pad.left;
    const legendGap = 14;
    legendItems.forEach(({ color, label }) => {
        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY, 14, 2);
        ctx.fillStyle = '#94A3B8';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(label, legendX + 18, legendY - 2);
        legendX += ctx.measureText(label).width + 34 + legendGap;
    });
}

/**
 * Load feature importance
 */
function loadFeatureImportance() {
    if (!modelInfo || !modelInfo.feature_importance) return;

    const tbody = document.getElementById('featureImportanceBody');
    const importance = modelInfo.feature_importance.ranked.slice(0, 10);

    tbody.innerHTML = importance
        .map(
            (feat, idx) => `
        <tr>
            <td>${feat.rank}</td>
            <td>${feat.name}</td>
            <td>
                <div class="bar-fill" style="width: ${feat.importance_pct}%"></div>
                ${feat.importance_pct}%
            </td>
        </tr>
    `
        )
        .join('');
}

/**
 * Export data
 */
function exportData() {
    const headers = ['Date', 'Volatility Daily', 'Volatility Annualized'];
    const rows = (tickerData.history || []).map(h => [
        h.date,
        formatNumber(h.volatility_daily),
        formatNumber(h.volatility_annualized),
    ]);

    downloadCSV(`${tickerData.ticker}-volatility-history.csv`, headers, rows);
    showToast('Datos exportados', 'success');
}

/**
 * Export chart as image
 */
function exportChart() {
    const canvas = document.getElementById('historyChart');
    if (!canvas) return;

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = `${tickerData.ticker}-volatility-chart.png`;
    link.click();

    showToast('Gráfico exportado', 'success');
}

