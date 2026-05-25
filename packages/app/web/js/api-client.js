/**
 * API Client for S&P 500 Volatility Predictor
 */

class APIClient {
    constructor(baseURL = CONFIG.API_BASE_URL) {
        this.baseURL = baseURL;
    }

    /**
     * Make HTTP request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error: ${endpoint}`, error);
            throw error;
        }
    }

    /**
     * GET /api/health
     * Check API health and model status
     */
    async health() {
        return this.request('/health');
    }

    /**
     * GET /api/screener/summary
     * Get market volatility summary
     */
    async screenerSummary() {
        const cached = cache.get('screener_summary');
        if (cached) return cached;

        const data = await this.request('/screener/summary');
        cache.set('screener_summary', data);
        return data;
    }

    /**
     * GET /api/screener
     * Get paginated screener data
     */
    async screener(options = {}) {
        const {
            limit = 50,
            offset = 0,
            sort = 'desc',
            sort_by = 'volatility',
            search = '',
            min_vol = 0,
            max_vol = 1,
        } = options;

        const params = new URLSearchParams({
            limit,
            offset,
            sort,
            sort_by,
            search,
            min_vol,
            max_vol,
        });

        return this.request(`/screener?${params.toString()}`);
    }

    /**
     * GET /api/predict/{ticker}
     * Get prediction and history for a ticker
     */
    async predict(ticker, options = {}) {
        const { period = settings.period, history_limit = 252 } = options;

        const params = new URLSearchParams({ period, history_limit });
        return this.request(`/predict/${ticker}?${params.toString()}`);
    }

    /**
     * GET /api/model/info
     * Get detailed model information
     */
    async modelInfo() {
        const cached = cache.get('model_info');
        if (cached) return cached;

        const data = await this.request('/model/info');
        cache.set('model_info', data);
        return data;
    }
}

// Global API client instance
const api = new APIClient();

/**
 * Handle API errors with user feedback
 */
function handleAPIError(error) {
    console.error('API Error:', error);

    if (error.message.includes('404')) {
        showToast('Recurso no encontrado (404)', 'error');
    } else if (error.message.includes('503')) {
        showToast('Servidor no disponible (503)', 'error');
    } else if (error.message.includes('not found')) {
        showToast('Ticker no encontrado', 'error');
    } else {
        showToast(`Error: ${error.message}`, 'error');
    }
}

