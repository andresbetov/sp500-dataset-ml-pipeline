/**
 * Common utilities for all pages
 */

const CONFIG = {
    API_BASE_URL: 'http://localhost:8080/api',
    CACHE_TTL: 5 * 60 * 1000, // 5 minutes
    DECIMAL_PLACES: 4,
    DEFAULT_PERIOD: '6mo',
};

let settings = {
    theme: localStorage.getItem('theme') || 'dark',
    decimalPlaces: parseInt(localStorage.getItem('decimalPlaces') || '4'),
    period: localStorage.getItem('period') || CONFIG.DEFAULT_PERIOD,
};

/**
 * Format a number to specified decimal places
 */
function formatNumber(num, decimals = settings.decimalPlaces) {
    if (typeof num !== 'number' || isNaN(num)) return '--';
    return num.toFixed(decimals);
}

/**
 * Format percentage change
 */
function formatPercentage(pct) {
    if (typeof pct !== 'number' || isNaN(pct)) return '--';
    const sign = pct >= 0 ? '▲' : '▼';
    return `${sign} ${Math.abs(pct).toFixed(2)}%`;
}

/**
 * Format large numbers with K, M, B suffix
 */
function formatLargeNumber(num) {
    if (typeof num !== 'number' || isNaN(num)) return '--';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
}

/**
 * Format date as YYYY-MM-DD
 */
function formatDate(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    return date.toISOString().split('T')[0];
}

/**
 * Format time ago (e.g., "5 minutes ago")
 */
function timeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'justo ahora';
    if (seconds < 3600) return `hace ${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `hace ${Math.floor(seconds / 3600)}h`;
    return `hace ${Math.floor(seconds / 86400)}d`;
}

/**
 * Open modal
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
}

/**
 * Close modal
 */
function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    }
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: ${type === 'success' ? '#10B981' : type === 'error' ? '#EF4444' : '#1E40AF'};
        color: white;
        padding: 16px 20px;
        border-radius: 8px;
        z-index: 2000;
        animation: slideIn 0.3s ease-out;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
}

/**
 * Debounce function
 */
function debounce(func, wait = 300) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, wait = 1000) {
    let timeout;
    return function executedFunction(...args) {
        if (!timeout) {
            func(...args);
            timeout = setTimeout(() => {
                timeout = null;
            }, wait);
        }
    };
}

/**
 * Local cache implementation
 */
class LocalCache {
    set(key, value, ttl = CONFIG.CACHE_TTL) {
        const data = {
            value,
            timestamp: Date.now(),
            ttl,
        };
        localStorage.setItem(`cache_${key}`, JSON.stringify(data));
    }

    get(key) {
        const data = localStorage.getItem(`cache_${key}`);
        if (!data) return null;

        const { value, timestamp, ttl } = JSON.parse(data);
        if (Date.now() - timestamp > ttl) {
            localStorage.removeItem(`cache_${key}`);
            return null;
        }
        return value;
    }

    clear(key) {
        localStorage.removeItem(`cache_${key}`);
    }

    clearAll() {
        Object.keys(localStorage).forEach(key => {
            if (key.startsWith('cache_')) {
                localStorage.removeItem(key);
            }
        });
    }
}

const cache = new LocalCache();

/**
 * Download data as CSV
 */
function downloadCSV(filename, headers, rows) {
    const csv = [
        headers.join(','),
        ...rows.map(row => row.map(cell => `"${cell}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

/**
 * Download JSON data
 */
function downloadJSON(filename, data) {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}

/**
 * Setup global event listeners
 */
function setupGlobalListeners() {
    // Settings button
    const settingsBtn = document.getElementById('settingsBtn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', () => openModal('settingsModal'));
    }

    // Health button
    const healthBtn = document.getElementById('healthBtn');
    if (healthBtn) {
        healthBtn.addEventListener('click', () => openModal('healthModal'));
    }

    // Global search
    const globalSearch = document.getElementById('globalSearch');
    if (globalSearch) {
        globalSearch.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const ticker = globalSearch.value.toUpperCase().trim();
                if (ticker) {
                    window.location.href = `ticker.html?ticker=${ticker}`;
                }
            }
        });

        globalSearch.addEventListener('input', debounce(async (e) => {
            const query = e.target.value.toUpperCase().trim();
            if (query.length < 1) {
                document.getElementById('searchSuggestions').classList.add('hidden');
                return;
            }

            // Could integrate with API for real search suggestions
            const suggestions = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'].filter(
                t => t.includes(query)
            );

            const suggestionsList = document.getElementById('searchSuggestions');
            if (suggestions.length > 0) {
                suggestionsList.innerHTML = suggestions
                    .map(
                        t =>
                            `<div class="search-suggestion-item" onclick="window.location.href='ticker.html?ticker=${t}'">${t}</div>`
                    )
                    .join('');
                suggestionsList.classList.remove('hidden');
            } else {
                suggestionsList.classList.add('hidden');
            }
        }, 200));
    }

    // Click outside modal to close
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) {
            e.target.classList.add('hidden');
            document.body.style.overflow = 'auto';
        }
    });

    // Save settings
    const saveSetting = document.getElementById('saveSetting');
    if (saveSetting) {
        saveSetting.addEventListener('click', () => {
            const decimalPlaces = document.getElementById('decimalPlaces')?.value;
            const period = document.getElementById('defaultPeriod')?.value;

            if (decimalPlaces) {
                localStorage.setItem('decimalPlaces', decimalPlaces);
                settings.decimalPlaces = parseInt(decimalPlaces);
            }
            if (period) {
                localStorage.setItem('period', period);
                settings.period = period;
            }

            showToast('Configuración guardada', 'success');
            closeModal('settingsModal');
        });
    }
}

/**
 * Get URL query parameters
 */
function getQueryParams() {
    const params = new URLSearchParams(window.location.search);
    const obj = {};
    for (let [key, value] of params) {
        obj[key] = value;
    }
    return obj;
}

/**
 * Set URL query parameters
 */
function setQueryParams(params) {
    const query = new URLSearchParams(params);
    window.history.pushState({}, '', `?${query.toString()}`);
}

/**
 * Regime label and color
 */
function getRegimeLabel(id) {
    const labels = {
        0: 'Pre-Crisis',
        1: 'Financial Crisis',
        2: 'Post-Crisis',
        3: 'COVID',
        4: 'Post-COVID',
    };
    return labels[id] || 'Unknown';
}

function getRegimeColor(id) {
    const colors = {
        0: '#3b82f6',
        1: '#ef4444',
        2: '#3b82f6',
        3: '#f97316',
        4: '#10b981',
    };
    return colors[id] || '#6b7280';
}

/**
 * Create badge element
 */
function createBadge(text, className = '') {
    const badge = document.createElement('span');
    badge.className = `badge ${className}`;
    badge.textContent = text;
    return badge;
}

/**
 * Setup page on DOMContentLoaded
 */
document.addEventListener('DOMContentLoaded', () => {
    setupGlobalListeners();

    // Load settings
    const decimalPlacesInput = document.getElementById('decimalPlaces');
    const periodSelect = document.getElementById('defaultPeriod');
    const themeSelect = document.getElementById('themeSelect');

    if (decimalPlacesInput) decimalPlacesInput.value = settings.decimalPlaces;
    if (periodSelect) periodSelect.value = settings.period;
    if (themeSelect) themeSelect.value = settings.theme;
});

