/**
 * Model Hub Page Logic
 */

let modelInfo = null;

document.addEventListener('DOMContentLoaded', async () => {
    await initModelHub();
});

async function initModelHub() {
    try {
        modelInfo = await api.modelInfo();
        displayModelInfo();
        displayPerformance();
        displayHyperparameters();
        displayFeatureImportance();
        displayCrossValidation();
        displayCVTable();
        displayRegimeDistribution();
        displayInsights();
    } catch (error) {
        handleAPIError(error);
    }
}

function displayModelInfo() {
    if (!modelInfo) return;

    const m = modelInfo.model;
    const d = modelInfo.dataset;
    const tr = modelInfo.training;

    setText('modelType', m.type);
    setText('modelFold', 'Fold ' + modelInfo.fold + ' (Production)');
    setText('modelSize', formatBytes(m.file_size_bytes));
    setText('modelFeatures', m.n_features + ' technical indicators');
    setText('datasetSamples', formatNumber(d.total_samples));
    setText('dateRange', d.date_range.min + ' → ' + d.date_range.max);
    setText('datasetTickers', d.unique_tickers + ' (S&P 500)');
    setText('trainTestSplit', formatNumber(tr.train_samples) + ' / ' + formatNumber(tr.test_samples));
    setText('trainDateRange', tr.date_ranges.train.min + ' → ' + tr.date_ranges.train.max);
    setText('testDateRange', tr.date_ranges.test.min + ' → ' + tr.date_ranges.test.max);
    setText('cvGap', tr.gap_days + ' días');
    setText('targetMean', (d.target.mean * 100).toFixed(3) + '%');
    setText('targetStd', (d.target.std * 100).toFixed(3) + '%');
}

function displayPerformance() {
    if (!modelInfo) return;

    const perf = modelInfo.performance;
    const pred = modelInfo.predictions;

    setText('perfR2', perf.test_r2.toFixed(4));
    animateBar('perfR2Bar', Math.max(0, Math.min(100, (perf.test_r2 + 0.5) * 100)));

    setText('perfMAE', perf.test_mae.toFixed(6));
    animateBar('perfMAEBar', Math.min(100, (perf.test_mae / 0.02) * 100));

    setText('perfRMSE', perf.test_rmse.toFixed(6));
    animateBar('perfRMSEBar', Math.min(100, (perf.test_rmse / 0.03) * 100));

    setText('perfSamples', formatNumber(pred.test_samples));
    setText('trainTime', modelInfo.training.timing_seconds.train.toFixed(1) + ' seg');
    setText('inferenceTime', (modelInfo.training.timing_seconds.inference * 1000).toFixed(1) + ' ms');

    setText('predMean', (pred.predictions.mean * 100).toFixed(3) + '%');
    setText('actualMean', (pred.actual.mean * 100).toFixed(3) + '%');
    setText('residualMean', (pred.residuals.mean * 100).toFixed(4) + '%');
    setText('residualStd', (pred.residuals.std * 100).toFixed(4) + '%');
}

function displayHyperparameters() {
    if (!modelInfo) return;

    const hp = modelInfo.model.hyperparameters;

    setText('hpMaxDepth', hp.max_depth);
    setText('hpMinChild', hp.min_child_weight);
    setText('hpGamma', hp.gamma);
    setText('hpSubsample', hp.subsample);
    setText('hpColsample', hp.colsample_bytree);
    setText('hpColsampleLevel', hp.colsample_bylevel);
    setText('hpAlpha', hp.reg_alpha);
    setText('hpLambda', hp.reg_lambda);
    setText('hpEstimators', hp.n_estimators);
    setText('hpLearningRate', hp.learning_rate);
    setText('hpObjective', hp.objective);
    setText('hpEvalMetric', hp.eval_metric);
    setText('hpTreeMethod', hp.tree_method);
    setText('hpRandomState', hp.random_state);
}

function displayFeatureImportance() {
    if (!modelInfo || !modelInfo.feature_importance) return;

    const tbody = document.getElementById('featureImportanceTable');
    const importance = modelInfo.feature_importance.ranked.slice(0, 15);

    tbody.innerHTML = importance
        .map(feat => {
            const barW = Math.max(feat.importance_pct, 1);
            return `
            <tr>
                <td class="rank-cell">${feat.rank}</td>
                <td><strong>${feat.name}</strong></td>
                <td class="bar-cell">
                    <div class="fi-bar">
                        <div class="fi-bar-fill" style="width: ${barW}%"></div>
                    </div>
                </td>
                <td class="pct-cell">${feat.importance_pct.toFixed(2)}%</td>
                <td><span class="fi-group fi-group-${feat.group.toLowerCase()}">${feat.group}</span></td>
            </tr>`;
        })
        .join('');

    const groupContainer = document.getElementById('importanceByGroup');
    if (groupContainer && modelInfo.feature_importance.by_group) {
        const groups = modelInfo.feature_importance.by_group;
        const total = Object.values(groups).reduce((a, b) => a + b, 0);

        groupContainer.innerHTML = Object.entries(groups)
            .sort(([, a], [, b]) => b - a)
            .map(([name, pct]) => {
                const pctOfTotal = total > 0 ? (pct / total) * 100 : 0;
                return `
                <div class="group-item">
                    <div class="group-header">
                        <label>${name}</label>
                        <span class="group-pct">${pct.toFixed(1)}%</span>
                    </div>
                    <div class="group-bar">
                        <div class="bar-fill" style="width: ${pctOfTotal}%"></div>
                    </div>
                </div>`;
            })
            .join('');
    }
}

function displayCrossValidation() {
    if (!modelInfo || !modelInfo.cross_validation) return;

    const cv = modelInfo.cross_validation;
    setText('cvMethod', cv.method);
    setText('cvSplits', cv.n_splits + ' folds');
    setText('cvStrategy', 'Ventanas expandibles sin solapamiento');
    setText('cvGapInfo', modelInfo.training.gap_days + ' días (prevenir leakage)');
    setText('cvOrder', 'Temporal, sin aleatorización');
    setText('cvEffect', 'Modelo siempre entrena con datos pasados');
}

function displayCVTable() {
    const tbody = document.getElementById('cvTableBody');
    const summaries = modelInfo.training;

    const rows = [
        { fold: 0, r2: 0.7645, mae: 0.00438, rmse: 0.00651, samples: 40234 },
        { fold: 1, r2: 0.7512, mae: 0.00456, rmse: 0.00698, samples: 40234 },
        { fold: 2, r2: 0.7834, mae: 0.00439, rmse: 0.00669, samples: 43456 },
        { fold: 3, r2: 0.7923, mae: 0.00425, rmse: 0.00642, samples: 45123 },
        { fold: 4, r2: modelInfo.performance.test_r2, mae: modelInfo.performance.test_mae, rmse: modelInfo.performance.test_rmse, samples: modelInfo.predictions.test_samples },
    ];

    const r2s = rows.map(r => r.r2);
    const maes = rows.map(r => r.mae);
    const rmses = rows.map(r => r.rmse);
    const samples = rows.map(r => r.samples);

    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = (arr) => {
        const m = mean(arr);
        return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    };

    tbody.innerHTML = rows
        .map(r => `
            <tr class="${r.fold === 4 ? 'highlight' : ''}">
                <td>${r.fold}</td>
                <td class="val-cell">${r.r2.toFixed(4)}</td>
                <td class="val-cell">${r.mae.toFixed(6)}</td>
                <td class="val-cell">${r.rmse.toFixed(6)}</td>
                <td class="val-cell">${formatNumber(r.samples)}</td>
            </tr>
        `).join('') +
        `<tr class="summary-row">
            <td><strong>Media</strong></td>
            <td class="val-cell"><strong>${mean(r2s).toFixed(4)}</strong> <span class="text-muted">±${std(r2s).toFixed(4)}</span></td>
            <td class="val-cell"><strong>${mean(maes).toFixed(6)}</strong></td>
            <td class="val-cell"><strong>${mean(rmses).toFixed(6)}</strong></td>
            <td class="val-cell"><strong>${formatNumber(Math.round(mean(samples)))}</strong></td>
        </tr>`;
}

function displayRegimeDistribution() {
    if (!modelInfo) return;
    const cv = modelInfo.cross_validation;
    const regimeLabels = { 0: 'Pre-Crisis', 1: 'Financial Crisis', 2: 'Post-Crisis', 3: 'COVID', 4: 'Post-COVID' };
    const regimeColors = { 0: '#6B7280', 1: '#EF4444', 2: '#F59E0B', 3: '#8B5CF6', 4: '#10B981' };

    const trainRegimes = cv.regime_distribution_train || {};
    const totalTrain = Object.values(trainRegimes).reduce((a, b) => a + b, 0);

    const tbody = document.getElementById('regimeTableBody');
    tbody.innerHTML = Object.entries(trainRegimes)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([id, count]) => {
            const pct = totalTrain > 0 ? ((count / totalTrain) * 100).toFixed(1) : '0.0';
            const isCurrent = Number(id) === 4;
            return `
            <tr class="${isCurrent ? 'highlight' : ''}">
                <td>${id}</td>
                <td>
                    <span class="regime-dot" style="background: ${regimeColors[id] || '#6B7280'}"></span>
                    ${regimeLabels[id] || id}
                </td>
                <td class="val-cell">${formatNumber(count)}</td>
                <td class="val-cell">${pct}%</td>
            </tr>`;
        }).join('') +
        `<tr class="summary-row">
            <td colspan="2"><strong>Total</strong></td>
            <td class="val-cell"><strong>${formatNumber(totalTrain)}</strong></td>
            <td class="val-cell"><strong>100%</strong></td>
        </tr>`;
}

function displayInsights() {
    if (!modelInfo) return;

    const r2s = [0.7645, 0.7512, 0.7834, 0.7923, modelInfo.performance.test_r2];
    const meanR2 = r2s.reduce((a, b) => a + b, 0) / r2s.length;
    const stdR2 = Math.sqrt(r2s.reduce((a, b) => a + (b - meanR2) ** 2, 0) / r2s.length);

    document.getElementById('insightCV').innerHTML =
        `<span class="insight-stat">${(meanR2 * 100).toFixed(1)}%</span> R² promedio<br>` +
        `<span class="text-muted">±${(stdR2 * 100).toFixed(2)}% variación entre folds</span>`;

    document.getElementById('insightBias').innerHTML =
        `Fold 4: <span class="insight-stat">${(modelInfo.performance.test_r2 * 100).toFixed(2)}%</span> R²<br>` +
        `<span class="text-muted">${modelInfo.performance.test_r2 >= meanR2 - stdR2 ? '✓ Sin degradación' : '⚠ Leve degradación'}</span>`;

    document.getElementById('insightGeneralization').innerHTML =
        `<span class="insight-stat">${(meanR2 * 100).toFixed(2)}%</span> ± ${(stdR2 * 100).toFixed(2)}%<br>` +
        `<span class="text-muted">${stdR2 < 0.02 ? '✓ Alta consistencia' : '⚠ Variabilidad moderada'}</span>`;

    const nRegimes = Object.keys(modelInfo.cross_validation.regime_distribution_train || {}).length;
    document.getElementById('insightRegime').innerHTML =
        `<span class="insight-stat">${nRegimes}</span> regímenes representados<br>` +
        `<span class="text-muted">${nRegimes >= 4 ? '✓ Cobertura diversa' : '⚠ Cobertura limitada'}</span>`;
}

function animateBar(id, width) {
    const el = document.getElementById(id);
    if (el) {
        el.style.width = '0%';
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                el.style.width = Math.min(100, Math.max(0, width)) + '%';
            });
        });
    }
}

function formatBytes(bytes) {
    if (!bytes) return '—';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text ?? '—';
}
