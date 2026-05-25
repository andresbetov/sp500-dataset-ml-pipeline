/**
 * Dashboard Page Logic
 */

let currentPage = 0;
const itemsPerPage = 10;
let marketSummary = null;
let topTickers = [];
let modelInfo = null;

document.addEventListener('DOMContentLoaded', async () => {
    await initDashboard();
});

/**
 * Initialize dashboard
 */
async function initDashboard() {
    try {
        updateHealthStatus();
        await Promise.all([
            loadMarketSummary(),
            loadTopTickers(),
            loadModelPerformance(),
        ]);

        document.getElementById('refreshMarket').addEventListener('click', async () => {
            cache.clear('screener_summary');
            cache.clear('model_info');
            showToast('Actualizando datos...', 'info');
            await Promise.all([
                loadMarketSummary(),
                loadTopTickers(),
                loadModelPerformance(),
            ]);
            showToast('Datos actualizados', 'success');
        });

        document.querySelectorAll('.vol-filter').forEach(cb => {
            cb.addEventListener('change', applyFilters);
        });
        document.querySelectorAll('.regime-filter').forEach(cb => {
            cb.addEventListener('change', applyFilters);
        });
    } catch (error) {
        handleAPIError(error);
    }
}

/**
 * Update health status indicator
 */
async function updateHealthStatus() {
    try {
        const health = await api.health();
        const healthStatus = document.getElementById('healthStatus');

        healthStatus.classList.add(health.status === 'ok' ? 'online' : 'offline');
        healthStatus.classList.remove(health.status === 'ok' ? 'offline' : 'online');

        document.getElementById('healthBtn').addEventListener('click', async () => {
            const content = document.getElementById('healthContent');
            content.innerHTML = `
                <div class="health-info">
                    <div class="health-grid">
                        <div class="health-item">
                            <span class="health-label">Estado</span>
                            <span class="health-value">
                                <span class="badge ${health.status === 'ok' ? 'badge-success' : 'badge-danger'}">
                                    ${health.status === 'ok' ? '● Online' : '○ Offline'}
                                </span>
                            </span>
                        </div>
                        <div class="health-item">
                            <span class="health-label">Modelo</span>
                            <span class="health-value">${health.model.type}</span>
                        </div>
                        <div class="health-item">
                            <span class="health-label">Árboles</span>
                            <span class="health-value">${health.model.n_estimators.toLocaleString()}</span>
                        </div>
                        <div class="health-item">
                            <span class="health-label">Profundidad</span>
                            <span class="health-value">${health.model.max_depth}</span>
                        </div>
                        <div class="health-item">
                            <span class="health-label">Features</span>
                            <span class="health-value">${health.model.n_features}</span>
                        </div>
                        <div class="health-item">
                            <span class="health-label">Dataset</span>
                            <span class="health-value">
                                <span class="badge ${health.dataset.exists ? 'badge-success' : 'badge-danger'}">
                                    ${health.dataset.exists ? '✓ Existe' : '✗ No encontrado'}
                                </span>
                            </span>
                        </div>
                    </div>
                </div>
            `;
            openModal('healthModal');
        });
    } catch (error) {
        console.warn('Health check failed:', error);
        document.getElementById('healthStatus').classList.add('offline');
    }
}

/**
 * Load market summary data
 */
async function loadMarketSummary() {
    try {
        const summary = await api.screenerSummary();
        marketSummary = summary;

        document.getElementById('meanVol').textContent = (summary.mean_daily_volatility * 100).toFixed(2) + '%';
        document.getElementById('annualVol').textContent = (summary.mean_annualized_volatility * 100).toFixed(2) + '%';
        document.getElementById('volRange').textContent =
            (summary.min_daily_volatility * 100).toFixed(2) + '% - ' +
            (summary.max_daily_volatility * 100).toFixed(2) + '%';
        document.getElementById('percentile').textContent = (summary.median_daily_volatility * 100).toFixed(2) + '%';

        document.getElementById('meanVolSidebar').textContent = (summary.mean_daily_volatility * 100).toFixed(2) + '%';
        document.getElementById('annualVolSidebar').textContent = (summary.mean_annualized_volatility * 100).toFixed(2) + '%';
        document.getElementById('tickerCountSidebar').textContent = summary.total_tickers;

        const datasetDate = new Date(summary.dataset.date_range.max);
        document.getElementById('datasetDate').textContent = formatDate(summary.dataset.date_range.max);
        document.getElementById('datasetFresh').textContent = 'Actualizado: ' + timeAgo(datasetDate.getTime());

        // Regime distribution in sidebar
        const regimeBox = document.getElementById('regimeDistribution');
        if (regimeBox && summary.regime_distribution) {
            regimeBox.innerHTML = Object.entries(summary.regime_distribution)
                .sort(([a], [b]) => Number(b) - Number(a))
                .map(([id, r]) => `
                    <div class="regime-row">
                        <span class="regime-dot regime-${id}"></span>
                        <span class="regime-label">${r.label}</span>
                        <span class="regime-count">${r.count}</span>
                        <span class="regime-pct">${r.pct}%</span>
                    </div>
                `).join('');
        }

        createDistributionChart(summary.percentiles);

        return summary;
    } catch (error) {
        document.getElementById('meanVol').textContent = '✕ Error';
        handleAPIError(error);
    }
}

/**
 * Load top tickers
 */
async function loadTopTickers() {
    try {
        const screener = await api.screener({ limit: 10, sort: 'desc' });
        topTickers = screener.tickers || [];
        renderTopTickers();
    } catch (error) {
        document.getElementById('topTickersBody').innerHTML =
            '<tr><td colspan="6" class="loading">Error al cargar datos</td></tr>';
        handleAPIError(error);
    }
}

function renderTopTickers() {
    const tbody = document.getElementById('topTickersBody');
    if (!topTickers.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading">Sin datos disponibles</td></tr>';
        return;
    }

    const regimeLabels = { 0: 'pre-crisis', 1: 'financial-crisis', 2: 'post-crisis', 3: 'covid', 4: 'post-covid' };

    tbody.innerHTML = topTickers
        .map((t, idx) => `
            <tr onclick="window.location.href='ticker.html?ticker=${t.ticker}'">
                <td class="rank">${idx + 1}</td>
                <td><strong class="ticker-name">${t.ticker}</strong></td>
                <td class="vol-cell">${(t.volatility_daily * 100).toFixed(2)}%</td>
                <td class="vol-cell">${(t.volatility_annualized * 100).toFixed(2)}%</td>
                <td>${t.last_price ? '$' + Number(t.last_price).toFixed(2) : '—'}</td>
                <td><span class="regime-badge regime-${t.regime}">${regimeLabels[t.regime] || '—'}</span></td>
            </tr>
        `).join('');
}

/**
 * Load model performance data from API
 */
async function loadModelPerformance() {
    try {
        const info = await api.modelInfo();
        modelInfo = info;

        const r2 = info.performance.test_r2;
        const mae = info.performance.test_mae;
        const rmse = info.performance.test_rmse;
        const nSamples = info.dataset.n_test_samples;

        document.getElementById('perfR2').textContent = r2.toFixed(4);
        document.getElementById('perfR2Bar').style.width = Math.max(0, Math.min(100, (r2 + 0.5) * 100)) + '%';

        document.getElementById('perfMAE').textContent = mae.toFixed(6);
        document.getElementById('perfMAEBar').style.width = Math.min(100, (mae / 0.02) * 100) + '%';

        document.getElementById('perfRMSE').textContent = rmse.toFixed(6);
        document.getElementById('perfRMSEBar').style.width = Math.min(100, (rmse / 0.03) * 100) + '%';

        document.getElementById('perfSamples').textContent = nSamples.toLocaleString();

        document.getElementById('perfFold').textContent = 'Fold ' + info.fold;
        document.getElementById('perfFeatures').textContent = info.feature_importance.ranked.length + ' features';
        document.getElementById('perfEstimators').textContent = info.hyperparameters.n_estimators + ' trees';
    } catch (error) {
        // Fallback to static data if API fails
        console.warn('Model info unavailable, using static values:', error);
    }
}

/**
 * Apply filters
 */
function applyFilters() {
    const selectedVols = Array.from(document.querySelectorAll('.vol-filter:checked')).map(cb => cb.value);
    const selectedRegimes = Array.from(document.querySelectorAll('.regime-filter:checked')).map(cb => Number(cb.value));

    if (!topTickers.length) return;

    const filtered = topTickers.filter(t => {
        const volOk = selectedVols.length === 0 ||
            selectedVols.some(v => {
                if (v === 'extrema') return t.volatility_daily > 0.04;
                if (v === 'alta') return t.volatility_daily > 0.025 && t.volatility_daily <= 0.04;
                if (v === 'normal') return t.volatility_daily > 0.015 && t.volatility_daily <= 0.025;
                if (v === 'baja') return t.volatility_daily <= 0.015;
                return true;
            });
        const regimeOk = selectedRegimes.length === 0 || selectedRegimes.includes(t.regime);
        return volOk && regimeOk;
    });

    const tbody = document.getElementById('topTickersBody');
    if (!filtered.length) {
        tbody.innerHTML = '<tr><td colspan="6" class="loading">No hay tickers que coincidan con los filtros</td></tr>';
        return;
    }

    const regimeLabels = { 0: 'pre-crisis', 1: 'financial-crisis', 2: 'post-crisis', 3: 'covid', 4: 'post-covid' };
    tbody.innerHTML = filtered
        .map((t, idx) => `
            <tr onclick="window.location.href='ticker.html?ticker=${t.ticker}'">
                <td class="rank">${idx + 1}</td>
                <td><strong class="ticker-name">${t.ticker}</strong></td>
                <td class="vol-cell">${(t.volatility_daily * 100).toFixed(2)}%</td>
                <td class="vol-cell">${(t.volatility_annualized * 100).toFixed(2)}%</td>
                <td>${t.last_price ? '$' + Number(t.last_price).toFixed(2) : '—'}</td>
                <td><span class="regime-badge regime-${t.regime}">${regimeLabels[t.regime] || '—'}</span></td>
            </tr>
        `).join('');
}

/**
 * Create enhanced distribution chart
 */
function createDistributionChart(percentiles) {
    const canvas = document.getElementById('distributionChart');
    if (!canvas) return;

    const container = canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(rect.width, 300);
    const H = 280;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const data = [
        { label: 'p10', value: percentiles.p10 },
        { label: 'p25', value: percentiles.p25 },
        { label: 'p50', value: percentiles.p50 },
        { label: 'p75', value: percentiles.p75 },
        { label: 'p90', value: percentiles.p90 },
    ];

    const maxVal = Math.max(...data.map(d => d.value), 0.001);
    const pad = { top: 24, right: 20, bottom: 40, left: 16 };
    const cw = W - pad.left - pad.right;
    const ch = H - pad.top - pad.bottom;
    const barW = Math.min(52, (cw / data.length) * 0.55);
    const gap = (cw - barW * data.length) / (data.length + 1);

    // Background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.2)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (let i = 0; i <= 4; i++) {
        const y = pad.top + (ch * i) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();
    }
    ctx.setLineDash([]);

    // Bars
    data.forEach((item, idx) => {
        const x = pad.left + gap + idx * (barW + gap);
        const bh = (item.value / maxVal) * ch;
        const y = pad.top + ch - bh;

        const ratio = item.value / maxVal;
        const r = Math.round(30 + ratio * 180);
        const g = Math.round(160 - ratio * 130);
        const b = Math.round(200);

        const grad = ctx.createLinearGradient(x, y, x, pad.top + ch);
        grad.addColorStop(0, `rgb(${r}, ${g}, ${b})`);
        grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.3)`);

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, bh, [4, 4, 0, 0]);
        ctx.fill();

        // Glow top
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.15)`;
        ctx.beginPath();
        ctx.roundRect(x - 2, y - 2, barW + 4, 6, [4, 4, 0, 0]);
        ctx.fill();

        // Value label
        ctx.fillStyle = '#E2E8F0';
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText((item.value * 100).toFixed(2) + '%', x + barW / 2, y - 6);

        // X label
        ctx.fillStyle = '#94A3B8';
        ctx.font = '12px sans-serif';
        ctx.textBaseline = 'top';
        ctx.fillText(item.label, x + barW / 2, pad.top + ch + 8);
    });
}
