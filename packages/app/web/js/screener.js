/**
 * Screener Page Logic
 */

let currentPage = 0;
const itemsPerPage = 50;
let allTickers = [];
let filteredTickers = [];

document.addEventListener('DOMContentLoaded', async () => {
    await initScreener();
});

/**
 * Initialize screener
 */
async function initScreener() {
    try {
        // Load initial data
        await loadScreenerData();

        // Setup event listeners
        document.getElementById('tickerSearch').addEventListener('input', debounce(filterTickers, 300));
        document.getElementById('minVolInput').addEventListener('change', filterTickers);
        document.getElementById('maxVolInput').addEventListener('change', filterTickers);
        document.getElementById('limitSelect').addEventListener('change', async () => {
            currentPage = 0;
            await renderTable();
        });
        document.getElementById('sortBySelect').addEventListener('change', filterTickers);
        document.getElementById('sortOrderSelect').addEventListener('change', filterTickers);
        document.getElementById('clearFiltersBtn').addEventListener('click', clearFilters);
        document.getElementById('refreshBtn').addEventListener('click', async () => {
            cache.clear('screener_data');
            await loadScreenerData();
            showToast('Datos actualizados', 'success');
        });
        document.getElementById('exportCsvBtn').addEventListener('click', exportToCSV);

        // Pagination
        document.getElementById('prevBtn').addEventListener('click', () => {
            if (currentPage > 0) {
                currentPage--;
                renderTable();
            }
        });

        document.getElementById('nextBtn').addEventListener('click', () => {
            const limit = parseInt(document.getElementById('limitSelect').value);
            const maxPage = Math.ceil(filteredTickers.length / limit);
            if (currentPage < maxPage - 1) {
                currentPage++;
                renderTable();
            }
        });
    } catch (error) {
        handleAPIError(error);
    }
}

/**
 * Load screener data
 */
async function loadScreenerData() {
    try {
        const screener = await api.screener({ limit: 500 });
        allTickers = screener.tickers;
        filteredTickers = [...allTickers];
        currentPage = 0;
        document.getElementById('tickerCounter').textContent = `${allTickers.length} tickers`;
        renderTable();
    } catch (error) {
        handleAPIError(error);
    }
}

/**
 * Filter tickers based on applied filters
 */
function filterTickers() {
    const search = document.getElementById('tickerSearch').value.toUpperCase().trim();
    const minVol = parseFloat(document.getElementById('minVolInput').value) || 0;
    const maxVol = parseFloat(document.getElementById('maxVolInput').value) || 1;
    const sortBy = document.getElementById('sortBySelect').value;
    const sortOrder = document.getElementById('sortOrderSelect').value;

    filteredTickers = allTickers.filter(ticker => {
        const matchesSearch = !search || ticker.ticker.includes(search);
        const matchesVolRange =
            ticker.volatility_daily >= minVol && ticker.volatility_daily <= maxVol;
        return matchesSearch && matchesVolRange;
    });

    // Sort
    if (sortBy === 'volatility') {
        filteredTickers.sort((a, b) =>
            sortOrder === 'desc'
                ? b.volatility_daily - a.volatility_daily
                : a.volatility_daily - b.volatility_daily
        );
    } else if (sortBy === 'ticker') {
        filteredTickers.sort((a, b) =>
            sortOrder === 'desc'
                ? b.ticker.localeCompare(a.ticker)
                : a.ticker.localeCompare(b.ticker)
        );
    }

    currentPage = 0;
    renderTable();
}

/**
 * Clear all filters
 */
function clearFilters() {
    document.getElementById('tickerSearch').value = '';
    document.getElementById('minVolInput').value = '';
    document.getElementById('maxVolInput').value = '';
    document.getElementById('sortBySelect').value = 'volatility';
    document.getElementById('sortOrderSelect').value = 'desc';

    filteredTickers = [...allTickers];
    currentPage = 0;
    renderTable();
}

/**
 * Render table with current page data
 */
function renderTable() {
    const limit = parseInt(document.getElementById('limitSelect').value);
    const start = currentPage * limit;
    const end = start + limit;
    const pageData = filteredTickers.slice(start, end);

    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = pageData
        .map(
            (ticker, idx) => `
        <tr onclick="window.location.href='ticker.html?ticker=${ticker.ticker}'">
            <td>${start + idx + 1}</td>
            <td><strong>${ticker.ticker}</strong></td>
            <td class="font-mono">${formatNumber(ticker.volatility_daily)}</td>
            <td class="font-mono">${formatNumber(ticker.volatility_annualized)}</td>
            <td>$${formatNumber(ticker.last_price)}</td>
            <td>
                <span class="regime-${ticker.regime}">
                    ${getRegimeLabel(ticker.regime)}
                </span>
            </td>
            <td class="text-muted">${formatDate(ticker.last_date)}</td>
            <td>
                <a href="ticker.html?ticker=${ticker.ticker}" class="link-secondary">
                    Analizar →
                </a>
            </td>
        </tr>
    `
        )
        .join('');

    // Update pagination info
    document.getElementById('displayCount').textContent = pageData.length;
    document.getElementById('totalCount').textContent = filteredTickers.length;
    document.getElementById('pageInfo').textContent = `Página ${currentPage + 1} de ${Math.ceil(filteredTickers.length / limit) || 1}`;

    // Update button states
    document.getElementById('prevBtn').disabled = currentPage === 0;
    document.getElementById('nextBtn').disabled =
        currentPage >= Math.ceil(filteredTickers.length / limit) - 1;
}

/**
 * Export to CSV
 */
function exportToCSV() {
    const headers = ['Rank', 'Ticker', 'Vol Daily', 'Vol Annualized', 'Price', 'Regime', 'Last Date'];
    const rows = filteredTickers.map((ticker, idx) => [
        idx + 1,
        ticker.ticker,
        formatNumber(ticker.volatility_daily),
        formatNumber(ticker.volatility_annualized),
        formatNumber(ticker.last_price),
        getRegimeLabel(ticker.regime),
        ticker.last_date,
    ]);

    downloadCSV('sp500-volatility-screener.csv', headers, rows);
    showToast('CSV exportado', 'success');
}

