/**
 * Initializes the line chart for displaying Accuracy per Episode.
 * @param {string[]} hosts - Array of host names to pre-initialize datasets. (Optional, if hosts are known upfront)
 */
function initializeLineChartAccuracy(agent, hostsDataset = [], canvasEl) {
    //if (lineChartAccuracy) lineChartAccuracy.destroy();
    canvasEl.parent().css("height", "400px");
    if (hostsDataset.length === 0) {
        console.error("initializeLineChartAccuracy: hostsDataset is empty, chart will start without predefined host datasets.");
        return;
    }
    if (hostsDataset[0].label === 'agent') {
        hostsDataset[0].label = agent; // Set the label to agent name for single-agent mode
    }

    return new Chart(canvasEl, {
        type: 'line',
        data: {
            labels: [],
            datasets: hostsDataset,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Episode',
                    },
                    beginAtZero: true
                },
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                    },
                    // min: 0,
                    // max: 100,
                },
            },
            plugins: {
                title: {
                    display: true,
                    text: agent + ' Accuracy per Episode',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || 'Agent';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2) + '%';
                            }
                            return label;
                        }
                    }
                }
            }
        },
    });
}

function initializeLineChartReward(agent, hostsDataset = [], canvasEl) {
    canvasEl.parent().css("height", "400px");
    if (hostsDataset.length === 0) {
        console.error("initializeLineChartReward: hostsDataset is empty, chart will start without predefined host datasets.");
        return;
    }
    if (hostsDataset[0].label === 'agent') {
        hostsDataset[0].label = agent; // Set the label to agent name for single-agent mode
    }
    return new Chart(canvasEl, {
        type: 'line',
        data: {
            // Labels (X-axis) will be episode numbers, shared by all agents.
            labels: [],
            datasets: hostsDataset,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Episode',
                    },
                    beginAtZero: true
                },
                y: {
                    title: {
                        display: true,
                        text: 'Reward',
                    },
                    // min: 0,
                    // max: 100,
                },
            },
            plugins: {
                title: {
                    display: true,
                    text: agent + ' Reward per Episode',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || 'Agent';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                }
            }
        },
    });
}


function ensureChartRedraw(chart, expectedLen) {
    if (!chart) return;
    try {
        chart.resize();
        chart.update();
    } catch (e) {
        // ignore
    }

    // Quick retry if dataset length doesn't match expectation
    try {
        const actualLen = (chart.data && chart.data.datasets && chart.data.datasets[0] && chart.data.datasets[0].data)
            ? chart.data.datasets[0].data.length
            : 0;
        if (expectedLen && actualLen !== expectedLen) {
            setTimeout(() => {
                try { chart.resize(); chart.update(); } catch (e) { }
                console.log('ensureChartRedraw: retry performed', { expectedLen, actualLenAfterRetry: (chart.data && chart.data.datasets && chart.data.datasets[0] && chart.data.datasets[0].data) ? chart.data.datasets[0].data.length : 0 });
            }, 200);
        }
    } catch (e) {
        // ignore
    }
}


/**
 * Applies a restored {labels, datasets: {key: [values...]}} series onto a
 * chart, matching/creating datasets by label (one per host in multi-agent
 * mode, one per agent otherwise) — mirrors how updateLineChartAccuracy /
 * updateLineChartReward build datasets during live updates, so restored
 * charts look identical to charts that were never reloaded.
 */
function applyRestoredHostSeries(chart, hostShapedData, isAccuracy) {
    if (!chart || !hostShapedData || !Array.isArray(hostShapedData.labels) || hostShapedData.labels.length === 0) {
        return false;
    }
    const labels = hostShapedData.labels;
    let didApply = false;

    Object.keys(hostShapedData.datasets || {}).forEach(key => {
        const raw = hostShapedData.datasets[key];
        if (!Array.isArray(raw) || raw.length === 0) return;

        let dataset = chart.data.datasets.find(ds => ds.label === key);
        if (!dataset) {
            const colorPrefix = (isAccuracy ? 'acc' : 'rew') + (isMultiAgentMode ? key : '');
            const color = getAgentColor(colorPrefix);
            dataset = {
                label: key,
                data: [],
                borderColor: color,
                backgroundColor: addTransparency(color, 0.2),
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: color,
                tension: 0.1,
                fill: false,
            };
            chart.data.datasets.push(dataset);
        }
        dataset.data = labels.map((_, i) => {
            if (i >= raw.length) return null;
            const value = Number(raw[i]);
            return isAccuracy ? value * 100 : value;
        });
        didApply = true;
    });

    if (didApply) {
        chart.data.labels = labels;
        chart.update();
        ensureChartRedraw(chart, labels.length);
    }
    return didApply;
}

function restoreAgentsChartsFromRawData(rawChartData) {
    const payload = rawChartData || chartDataRaw;
    if (!payload || !currentConfig || !currentConfig.cfg) {
        return false;
    }

    const configuredAgents = currentConfig.cfg.agents || [];
    let didUpdate = false;

    configuredAgents.forEach(agent => {
        const isSupervised = String(agent).toLowerCase().includes('supervised');

        const accuracyRaw = payload.accuracy ? payload.accuracy[agent] : undefined;
        const rewardRaw = (!isSupervised && payload.reward) ? payload.reward[agent] : undefined;
        const accuracyChart = agentsChartAccuracy[agent];
        const rewardChart = isSupervised ? null : agentsChartReward[agent];

        // Logging context for this agent
        console.groupCollapsed(`restoreAgentsChartsFromRawData: agent='${agent}'`);
        if (!accuracyChart) {
            if (currentConfig.cfg && currentConfig.cfg.agents && currentConfig.cfg.hosts) {
                initializeAgentsCharts(currentConfig.cfg.isMultiAgent, currentConfig.cfg.agents, currentConfig.cfg.hosts); // Initialize all charts with config data
            }
            console.log(' -> accuracyChart: not found, attempted re-initialization');
        }
        if (accuracyChart && accuracyRaw) {
            // mark tab as finished and open its charts if visible
            if (typeof setAgentMetricsTabStatus === 'function') {
                setAgentMetricsTabStatus(agent, 'finished');
            }

            if (applyRestoredHostSeries(accuracyChart, accuracyRaw, true)) {
                console.log(' -> accuracy: restored');
                didUpdate = true;
            } else {
                console.log(' -> accuracy: skipped (empty series)');
            }
        }
        else if (accuracyChart) {
            console.log(' -> accuracy: skipped (null/undefined from payload)');
        }
        else if (accuracyRaw) {
            console.log(' -> accuracy: skipped (null/undefined from chart instance)');
        }

        if (rewardChart && rewardRaw) {
            if (applyRestoredHostSeries(rewardChart, rewardRaw, false)) {
                console.log(' -> reward: restored');
                didUpdate = true;
            } else {
                console.log(' -> reward: skipped (empty series)');
            }
        }
        else if (rewardChart) {
            console.log(' -> reward: skipped (null/undefined from payload)');
        }
        else if (rewardRaw) {
            console.log(' -> reward: skipped (null/undefined from chart instance)');
        }

        console.groupEnd();
    });

    // Activate the first agent tab that has restored data
    if (didUpdate && typeof activateAgentMetricsTab === 'function') {
        const firstRestored = configuredAgents.find(a => {
            const hasAcc = payload.accuracy && payload.accuracy[a] !== undefined;
            const hasRew = payload.reward  && payload.reward[a]  !== undefined;
            return hasAcc || hasRew;
        });
        if (firstRestored) {
            activateAgentMetricsTab(firstRestored);
        }
    }

    return didUpdate;
}


function initializeAgentsCharts(isMultiAgent, configuredAgents, configuredHosts) {
    isMultiAgentMode = isMultiAgent;
    if (!configuredAgents || configuredAgents.length === 0) {
        console.warn("No agents configured for chart initialization.");
        return;
    }
    if (!configuredHosts || configuredHosts.length === 0) {
        console.warn("No hosts configured for chart initialization.");
        return;
    }

    const configuredRenderableAgents = configuredAgents;
    const existingAgentSections = $("#charts-metrics-section div.agent-section[data-agent]").length;
    const existingCharts = Object.keys(agentsChartAccuracy).length > 0 || Object.keys(agentsChartReward).length > 0;
    const alreadyBuilt = existingAgentSections === configuredRenderableAgents.length
        && existingCharts
        && agents.length === configuredAgents.length
        && hosts.length === configuredHosts.length;
    if (alreadyBuilt) {
        if (typeof applyPendingChartRestore === 'function') {
            applyPendingChartRestore();
        }
        if (typeof restoreTrainingResultsButtonStates === 'function') {
            restoreTrainingResultsButtonStates();
        }
        if (typeof restoreEvaluationResultsButtonStates === 'function') {
            restoreEvaluationResultsButtonStates();
        }
        return;
    }

    for (const agent in agentsChartAccuracy) {
        if (agentsChartAccuracy[agent]) {
            agentsChartAccuracy[agent].destroy();
            delete agentsChartAccuracy[agent];
        }
    }
    for (const agent in agentsChartReward) {
        if (agentsChartReward[agent]) {
            agentsChartReward[agent].destroy();
            delete agentsChartReward[agent];
        }
    }
    agentsChartAccuracy = {};
    agentsChartReward = {};

    $("#charts-metrics-section div.agent-section[data-agent]").remove();

    chartMetricsSection.removeClass('hidden');
    agents = configuredAgents; // Store agents globally for later use
    hosts = configuredHosts;
    // for (let i = 0; i < agents.length; i++) {
    datasetForAccuracy = [];
    datasetForReward = [];
    if (!isMultiAgentMode) {
        const colorAcc = getAgentColor("acc");
        datasetForAccuracy.push(getDataset("agent", colorAcc));
        const colorRew = getAgentColor("rew");
        datasetForReward.push(getDataset("agent", colorRew));
    }
    else {
        for (let i = 0; i < hosts.length; i++) {
            const colorAcc = getAgentColor("acc" + hosts[i]);
            datasetForAccuracy.push(getDataset(hosts[i], colorAcc));
            const colorRew = getAgentColor("rew" + hosts[i]);
            datasetForReward.push(getDataset(hosts[i], colorRew));
        }
    }
    for (let i = 0; i < agents.length; i++) {
        const isSupervised = agents[i].toLowerCase().includes("supervised");

        // Clone the agent section template
        let agentSection = $("div.agent-section").first().clone();

        // Remove hidden class and set agent identifier
        agentSection.removeClass("hidden");
        agentSection.attr("data-agent", agents[i]);

        agentSection.find(".agent-title").html(getAgentSummaryHeaderHtml(agents[i]));
        $("#charts-metrics-section").append(agentSection);

        // Prepare datasets for this agent
        let datasetForAccuracyAgent = structuredClone(datasetForAccuracy);

        // Get the canvas elements within this agent section
        let accuracyCanvas = agentSection.find("canvas.lineChartAccuracy").first();

        // Initialize accuracy chart for all agents
        agentsChartAccuracy[agents[i]] = initializeLineChartAccuracy(agents[i], datasetForAccuracyAgent, accuracyCanvas);

        if (isSupervised) {
            // Hide reward chart and host data section for supervised agents
            agentSection.find(".reward-chart-title").closest("details").addClass("hidden");
            agentSection.find(".agent-data-section").addClass("hidden");
            continue;
        }

        // Reward chart only for non-supervised agents
        let datasetForRewardAgent = structuredClone(datasetForReward);
        let rewardCanvas = agentSection.find("canvas.lineChartReward").first();
        agentsChartReward[agents[i]] = initializeLineChartReward(agents[i], datasetForRewardAgent, rewardCanvas);

        let agentDataSection = agentSection.find(".agent-data-section").first();
        //add an UL to agentDataSection
        agentDataSection.html(""); // Clear existing content
        let ulElement = $("<ul></ul>");
        agentDataSection.append(ulElement);

        // this is when we use multi agents.
        if (isMultiAgentMode) {
            let coordinatorAgentKey = `${agents[i]}_${COORDINATOR}`;
            let ilElement = getLiElement(coordinatorAgentKey, COORDINATOR, false);
            ulElement.append(ilElement);
            for (let j = 0; j < hosts.length; j++) {
                let host = hosts[j];
                let hostAgentKey = `${agents[i]}_${host}`;
                let ilElement = getLiElement(hostAgentKey, host, true);
                ulElement.append(ilElement);
            }
        }
        else if (isSingleAgentHostObservableEnv()) {
            for (let j = 0; j < hosts.length; j++) {
                let host = hosts[j];
                let hostAgentKey = `${agents[i]}_${host}`;
                let ilElement = getLiElement(hostAgentKey, host, true);
                ulElement.append(ilElement);
            }
        }
        else {
            let agentKey = agents[i];
            let ilElement = getLiElement(agentKey, "", false);
            ulElement.append(ilElement);
        }
    }

    // === Build tab bar ===
    // Preserve statuses from a previous call (e.g. socket reconnect or the
    // reload-recovery flow both re-run this) so an agent already known to be
    // 'running'/'finished' doesn't visually fall back to 'waiting'.
    const previousTabStatuses = agentTabStatuses;
    agentTabStatuses = {};
    const tabBar = $('#agent-metrics-tab-bar');
    tabBar.empty();
    const isSequential = isSequentialMarlEnv();

    for (let i = 0; i < agents.length; i++) {
        const agentName = agents[i];
        const initialStatus = previousTabStatuses[agentName]
            || (isSequential ? 'waiting' : 'running');
        agentTabStatuses[agentName] = initialStatus;

        const isActive = (i === 0);
        const badgeClass = _agentTabBadgeClasses(initialStatus);
        const badgeText  = _agentTabBadgeText(initialStatus);
        const activeTabClass = isActive
            ? 'border-blue-600 text-blue-700 bg-white'
            : 'border-transparent text-gray-500';

        const tabBtn = $(`
            <button type="button"
                class="agent-metrics-tab flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap transition-colors hover:text-gray-700 ${activeTabClass}"
                data-agent="${escapeAgentSummaryHtml(agentName)}">
                <span class="agent-tab-name">${escapeAgentSummaryHtml(agentName)}</span>
                <span class="agent-tab-status-badge inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${badgeClass}">${badgeText}</span>
            </button>
        `);
        tabBar.append(tabBtn);
    }

    if (agents.length > 0) {
        tabBar.addClass('flex').removeClass('hidden');
    }

    // Show only the first agent panel; for parallel scenarios open charts immediately
    for (let i = 0; i < agents.length; i++) {
        const agentName = agents[i];
        const section = $(`#charts-metrics-section .agent-section[data-agent="${agentName}"]`);
        if (i === 0) {
            section.removeClass('hidden');
            if (!isSequential) {
                section.find('.agent-charts-section details').prop('open', true);
            }
        } else {
            section.addClass('hidden');
        }
    }
    // === End tab bar ===

    // Reload/reconnect recovery: server told us which agent is currently
    // training/evaluating (see /get_training_status's current_agent field).
    // Applied here, after tabs exist, regardless of which caller triggered
    // this rebuild.
    if (window.pendingCurrentTrainingAgent) {
        applyCurrentAgentFromServer(window.pendingCurrentTrainingAgent);
    }

    if (typeof applyPendingChartRestore === 'function') {
        applyPendingChartRestore();
    }
    if (typeof restoreTrainingResultsButtonStates === 'function') {
        restoreTrainingResultsButtonStates();
    }
    if (typeof restoreEvaluationResultsButtonStates === 'function') {
        restoreEvaluationResultsButtonStates();
    }
}

// ====================================================================
// AGENT METRICS TAB MANAGEMENT
// ====================================================================

let agentTabStatuses = {}; // agentName → 'waiting' | 'running' | 'finished'

/**
 * Returns true for marl_* gym types where agents train sequentially.
 * Returns false for classification/attacks where agents run in parallel.
 */
function isSequentialMarlEnv() {
    const gymType = String(
        (currentConfig && currentConfig.env_params && currentConfig.env_params.gym_type) || ''
    ).toLowerCase();
    return gymType.startsWith('marl_') || gymType.startsWith('attacks_ho');
}

function _agentTabBadgeClasses(status) {
    if (status === 'running')  return 'bg-green-100 text-green-700';
    if (status === 'finished') return 'bg-blue-100 text-blue-700';
    return 'bg-gray-200 text-gray-600';
}

function _agentTabBadgeText(status) {
    if (status === 'running')  return 'Running';
    if (status === 'finished') return 'Finished';
    return 'Waiting';
}

/**
 * Switches the visible agent panel to agentName and highlights its tab button.
 */
function activateAgentMetricsTab(agentName) {
    // Update tab button styles
    $('#agent-metrics-tab-bar .agent-metrics-tab').each(function () {
        const isActive = $(this).data('agent') === agentName;
        $(this)
            .toggleClass('border-blue-600 text-blue-700 bg-white', isActive)
            .toggleClass('border-transparent text-gray-500', !isActive);
    });

    // Show the matching panel, hide all others
    $('#charts-metrics-section .agent-section[data-agent]').each(function () {
        const isActive = $(this).data('agent') === agentName;
        $(this).toggleClass('hidden', !isActive);
    });

    // If the agent has data (running or finished), open its charts
    const status = agentTabStatuses[agentName];
    if (status === 'running' || status === 'finished') {
        const panel = $(`#charts-metrics-section .agent-section[data-agent="${agentName}"]`);
        panel.find('.agent-charts-section details').prop('open', true);
    }
}

/**
 * Updates the status badge on the agent's tab button and
 * opens charts when the agent becomes active/finished.
 */
function setAgentMetricsTabStatus(agentName, status) {
    agentTabStatuses[agentName] = status;

    const tab = $(`#agent-metrics-tab-bar .agent-metrics-tab[data-agent="${agentName}"]`);
    if (!tab.length) return;

    const badge = tab.find('.agent-tab-status-badge');
    badge
        .removeClass('bg-gray-200 text-gray-600 bg-green-100 text-green-700 bg-blue-100 text-blue-700')
        .addClass(_agentTabBadgeClasses(status))
        .text(_agentTabBadgeText(status));

    // Open charts if the panel is currently visible
    if (status === 'running' || status === 'finished') {
        const panel = $(`#charts-metrics-section .agent-section[data-agent="${agentName}"]`);
        if (!panel.hasClass('hidden')) {
            panel.find('.agent-charts-section details').prop('open', true);
        }
    }
}

/**
 * Called from statistics.js whenever chart data arrives for an agent.
 * Transitions the agent from 'waiting' → 'running' and, in sequential MARL,
 * auto-activates that agent's tab.
 */
function notifyAgentChartUpdated(agentName) {
    const current = agentTabStatuses[agentName];
    if (!current || current === 'waiting') {
        setAgentMetricsTabStatus(agentName, 'running');
        if (isSequentialMarlEnv()) {
            activateAgentMetricsTab(agentName);
        }
    }
}

/**
 * Called after reload/reconnect recovery (/get_training_status) with the
 * agent the server says is currently training/evaluating. Live metric
 * messages can take a while to arrive for slow environments (e.g. Mininet
 * network-step simulation), so without this the tab would show 'Waiting'
 * even though the agent is actively running.
 */
function applyCurrentAgentFromServer(agentName) {
    if (!agentName || !(agentName in agentTabStatuses)) return;
    if (agentTabStatuses[agentName] === 'finished') return;
    setAgentMetricsTabStatus(agentName, 'running');
    if (isSequentialMarlEnv()) {
        activateAgentMetricsTab(agentName);
    }
}

/**
 * Resets all tab state (called on new training run).
 */
function resetAgentMetricsTabState() {
    agentTabStatuses = {};
    const tabBar = $('#agent-metrics-tab-bar');
    tabBar.empty().addClass('hidden').removeClass('flex');
}

// Tab click delegation
$(document).on('click', '#agent-metrics-tab-bar .agent-metrics-tab', function () {
    const agentName = $(this).data('agent');
    activateAgentMetricsTab(agentName);
});

// ====================================================================

function escapeAgentSummaryHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function getAgentSummaryHeaderHtml(agentName) {
    const escapedAgentName = escapeAgentSummaryHtml(agentName);
    return `
        <div class="flex items-center justify-between gap-3">
            <span>Agent: ${escapedAgentName}</span>
            <span class="flex items-center gap-2">
                <button
                    type="button"
                    class="agent-training-popup-btn inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-slate-200 text-slate-500 text-xs font-semibold cursor-not-allowed"
                    data-agent="${escapedAgentName}"
                    data-ready="false"
                    disabled
                    title="Training summary available when the agent finishes plotting">
                    <img src="/static/images/gif/training.gif" alt="Training result" class="w-4 h-4 rounded">
                    Training result
                </button>
                <button
                    type="button"
                    class="agent-evaluation-popup-btn inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-slate-200 text-slate-500 text-xs font-semibold cursor-not-allowed"
                    data-agent="${escapedAgentName}"
                    data-ready="false"
                    disabled
                    title="Evaluation summary available when the agent finishes testing">
                    <img src="/static/images/gif/test.gif" alt="Evaluation result" class="w-4 h-4 rounded">
                    Evaluation result
                </button>
            </span>
        </div>
    `;
}

function formatTrainingMetricValue(metricKey, value) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
        return 'N/A';
    }
    if (['accuracy', 'precision', 'recall', 'f1_score', 'qtable_coverage_pct', 'exploration_rate'].includes(metricKey)) {
        return `${(numericValue * 100).toFixed(2)}%`;
    }
    if (metricKey === 'policy_entropy' || metricKey === 'q_values_std' || metricKey === 'q_values_mean' || metricKey === 'q_values_max') {
        return numericValue.toFixed(4);
    }
    if (metricKey === 'cumulative_reward') {
        return numericValue.toFixed(2);
    }
    return numericValue.toFixed(2);
}

function sortTrainingCharts(chartFiles) {
    const priority = [
        'metrics_combined',
        'metrics',
        'rewards',
        'matrix',
        'qtable_coverage',
        'policy_exploration',
        'bin_coverage',
    ];
    return [...(Array.isArray(chartFiles) ? chartFiles : [])].sort((left, right) => {
        const leftName = String(left || '').toLowerCase();
        const rightName = String(right || '').toLowerCase();
        const leftIndex = priority.findIndex(token => leftName.includes(token));
        const rightIndex = priority.findIndex(token => rightName.includes(token));
        const normalizedLeft = leftIndex === -1 ? priority.length : leftIndex;
        const normalizedRight = rightIndex === -1 ? priority.length : rightIndex;
        if (normalizedLeft !== normalizedRight) {
            return normalizedLeft - normalizedRight;
        }
        return leftName.localeCompare(rightName);
    });
}

function renderMetricCards(summary) {
    const latestMetrics = summary && typeof summary.latest_metrics === 'object' && summary.latest_metrics !== null
        ? summary.latest_metrics
        : {};
    const cards = [
        ['Episodes', summary.episodes_completed ?? 0],
        ['Last Steps', summary.steps_last_episode ?? 0],
        ['Elapsed', Number.isFinite(Number(summary.train_execution_time)) ? `${Number(summary.train_execution_time).toFixed(1)}s` : 'N/A'],
        ['Charts', summary.chart_count ?? 0],
        ['Accuracy', formatTrainingMetricValue('accuracy', latestMetrics.accuracy)],
        ['Precision', formatTrainingMetricValue('precision', latestMetrics.precision)],
        ['Recall', formatTrainingMetricValue('recall', latestMetrics.recall)],
        ['F1', formatTrainingMetricValue('f1_score', latestMetrics.f1_score)],
    ];

    return cards.map(([label, value]) => `
        <div class="bg-white border border-slate-200 rounded-xl p-3">
            <div class="text-[11px] uppercase tracking-wide text-slate-500 font-semibold">${escapeAgentSummaryHtml(label)}</div>
            <div class="text-sm font-bold text-slate-500 mt-1">${escapeAgentSummaryHtml(value)}</div>
        </div>
    `).join('');
}

function renderExtraMetrics(summary) {
    const latestMetrics = summary && typeof summary.latest_metrics === 'object' && summary.latest_metrics !== null
        ? summary.latest_metrics
        : {};
    const metricLabels = {
        cumulative_reward: 'Cumulative Reward',
        qtable_coverage_pct: 'Q-table Coverage',
        exploration_rate: 'Exploration Rate',
        policy_entropy: 'Policy Entropy',
        q_values_std: 'Q-values Std',
        q_values_mean: 'Q-values Mean',
        q_values_max: 'Q-values Max',
    };

    const extraRows = Object.keys(metricLabels)
        .filter(metricKey => latestMetrics[metricKey] !== undefined && latestMetrics[metricKey] !== null)
        .map(metricKey => `
            <div class="flex items-center justify-between text-sm border-b border-slate-100 py-1.5">
                <span class="text-slate-300 font-medium">${escapeAgentSummaryHtml(metricLabels[metricKey])}</span>
                <span class="font-semibold text-slate-500">${escapeAgentSummaryHtml(formatTrainingMetricValue(metricKey, latestMetrics[metricKey]))}</span>
            </div>
        `)
        .join('');

    if (!extraRows) {
        return '';
    }

    return `
        <div class="bg-white border border-slate-200 rounded-xl p-4">
            <div class="text-xs font-bold uppercase text-slate-700 mb-2">Extra Training Signals</div>
            <div>${extraRows}</div>
        </div>
    `;
}

function renderPerHostMetrics(summary) {
    const perHostMetrics = Array.isArray(summary.per_host_metrics) ? summary.per_host_metrics : [];
    if (!perHostMetrics.length) {
        return '';
    }

    const rows = perHostMetrics.map(item => {
        const hostMetrics = item && typeof item.latest_metrics === 'object' ? item.latest_metrics : {};
        return `
            <tr class="border-b border-slate-100">
                <td class="p-2 font-semibold text-slate-700">${escapeAgentSummaryHtml(item.host || '-')}</td>
                <td class="p-2 text-right">${escapeAgentSummaryHtml(formatTrainingMetricValue('accuracy', hostMetrics.accuracy))}</td>
                <td class="p-2 text-right">${escapeAgentSummaryHtml(formatTrainingMetricValue('precision', hostMetrics.precision))}</td>
                <td class="p-2 text-right">${escapeAgentSummaryHtml(formatTrainingMetricValue('recall', hostMetrics.recall))}</td>
                <td class="p-2 text-right">${escapeAgentSummaryHtml(formatTrainingMetricValue('f1_score', hostMetrics.f1_score))}</td>
            </tr>
        `;
    }).join('');

    return `
        <div class="bg-white border border-slate-200 rounded-xl p-4">
            <div class="text-xs font-bold uppercase text-slate-700 mb-2">Per-host Training Snapshot</div>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-slate-200 text-slate-500 uppercase text-[11px]">
                            <th class="p-2 text-left">Host</th>
                            <th class="p-2 text-right">Accuracy</th>
                            <th class="p-2 text-right">Precision</th>
                            <th class="p-2 text-right">Recall</th>
                            <th class="p-2 text-right">F1</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>
    `;
}

function renderTrainingCharts(summary) {
    const relativeDir = String(summary.relative_dir || '');
    const chartFiles = sortTrainingCharts(summary.chart_files || []);

    if (!relativeDir || !chartFiles.length) {
        return `
            <div class="bg-white border border-slate-200 rounded-xl p-4 text-sm text-slate-500">
                No training charts available for this agent yet.
            </div>
        `;
    }

    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            ${chartFiles.map(chartFile => `
                <div class="bg-white border border-slate-200 rounded-xl p-2">
                    <div class="text-[11px] font-semibold uppercase text-slate-500 mb-2">${escapeAgentSummaryHtml(chartFile)}</div>
                    <img
                        src="/static-training/${encodeURI(relativeDir)}/${encodeURIComponent(chartFile)}"
                        alt="${escapeAgentSummaryHtml(chartFile)}"
                        title="${escapeAgentSummaryHtml(chartFile)}"
                        class="clickable-img w-full h-56 object-contain rounded-lg border border-slate-100 bg-slate-50 cursor-zoom-in"
                        data-description="${escapeAgentSummaryHtml(summary.agent_name || 'Agent')} - ${escapeAgentSummaryHtml(chartFile)}"
                    >
                </div>
            `).join('')}
        </div>
    `;
}

function renderEvaluationMetricCards(summary) {
    const latestMetrics = summary && typeof summary.latest_metrics === 'object' && summary.latest_metrics !== null
        ? summary.latest_metrics
        : {};
    const score = Number(summary.score);
    const testEpisodes = Number(summary.test_episodes || 0);
    const scorePct = testEpisodes > 0 && Number.isFinite(score)
        ? `${((score / testEpisodes) * 100).toFixed(2)}%`
        : 'N/A';
    const cards = [
        ['Test Episodes', summary.test_episodes ?? 0],
        ['Score', Number.isFinite(score) ? score : 'N/A'],
        ['Score %', scorePct],
        ['Charts', summary.chart_count ?? 0],
        ['Accuracy', formatTrainingMetricValue('accuracy', latestMetrics.accuracy)],
        ['Precision', formatTrainingMetricValue('precision', latestMetrics.precision)],
        ['Recall', formatTrainingMetricValue('recall', latestMetrics.recall)],
        ['F1', formatTrainingMetricValue('f1_score', latestMetrics.f1_score)],
    ];

    return cards.map(([label, value]) => `
        <div class="bg-white border border-amber-200 rounded-xl p-3">
            <div class="text-[11px] uppercase tracking-wide text-amber-700 font-semibold">${escapeAgentSummaryHtml(label)}</div>
            <div class="text-sm font-bold text-slate-500 mt-1">${escapeAgentSummaryHtml(value)}</div>
        </div>
    `).join('');
}

function renderEvaluationMitigation(summary) {
    const mitigationSummary = summary && typeof summary.mitigation_summary === 'object' && summary.mitigation_summary !== null
        ? summary.mitigation_summary
        : null;
    if (!mitigationSummary) {
        return '';
    }

    const ratioPct = Number(mitigationSummary.mitigated_under_attack_ratio || 0) * 100;
    return `
        <div class="bg-white border border-amber-200 rounded-xl p-4">
            <div class="text-xs font-bold uppercase text-amber-700 mb-2">Attack Mitigation</div>
            <div class="space-y-1 text-sm">
                <div class="flex items-center justify-between"><span class="text-slate-400">Episodes with data</span><span class="font-semibold text-slate-500">${escapeAgentSummaryHtml(mitigationSummary.episodes_with_mitigation_data || 0)}</span></div>
                <div class="flex items-center justify-between"><span class="text-slate-400">Under attack total</span><span class="font-semibold text-slate-500">${escapeAgentSummaryHtml(mitigationSummary.total_under_attack_count || 0)}</span></div>
                <div class="flex items-center justify-between"><span class="text-slate-400">Mitigated total</span><span class="font-semibold text-slate-500">${escapeAgentSummaryHtml(mitigationSummary.total_mitigated_under_attack_count || 0)}</span></div>
                <div class="flex items-center justify-between"><span class="text-slate-400">Mitigation ratio</span><span class="font-semibold text-slate-500">${escapeAgentSummaryHtml(`${ratioPct.toFixed(2)}%`)}</span></div>
            </div>
        </div>
    `;
}

function renderEvaluationCharts(summary) {
    const relativeDir = String(summary.relative_dir || '');
    const chartFiles = sortTrainingCharts(summary.chart_files || []);

    if (!relativeDir || !chartFiles.length) {
        return `
            <div class="bg-white border border-amber-200 rounded-xl p-4 text-sm text-slate-500">
                No evaluation charts available for this agent yet.
            </div>
        `;
    }

    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            ${chartFiles.map(chartFile => `
                <div class="bg-white border border-amber-200 rounded-xl p-2">
                    <div class="text-[11px] font-semibold uppercase text-amber-700 mb-2">${escapeAgentSummaryHtml(chartFile)}</div>
                    <img
                        src="/static-training/${encodeURI(relativeDir)}/${encodeURIComponent(chartFile)}"
                        alt="${escapeAgentSummaryHtml(chartFile)}"
                        title="${escapeAgentSummaryHtml(chartFile)}"
                        class="clickable-img w-full h-56 object-contain rounded-lg border border-amber-100 bg-amber-50 cursor-zoom-in"
                        data-description="${escapeAgentSummaryHtml(summary.agent_name || 'Agent')} - ${escapeAgentSummaryHtml(chartFile)}"
                    >
                </div>
            `).join('')}
        </div>
    `;
}

function renderAgentEvaluationSummaryPopup(summary) {
    return `
        <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                ${renderEvaluationMetricCards(summary)}
            </div>
            ${renderEvaluationMitigation(summary)}
            <div class="bg-amber-50 border border-amber-200 rounded-xl p-4">
                <div class="text-xs font-bold uppercase text-amber-700 mb-3">Evaluation Charts</div>
                ${renderEvaluationCharts(summary)}
            </div>
        </div>
    `;
}

function renderAgentTrainingSummaryPopup(summary) {
    return `
        <div class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
                ${renderMetricCards(summary)}
            </div>
            ${renderExtraMetrics(summary)}
            ${renderPerHostMetrics(summary)}
            <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                <div class="text-xs font-bold uppercase text-slate-700 mb-3">Training Charts</div>
                ${renderTrainingCharts(summary)}
            </div>
        </div>
    `;
}

function updateAgentTrainingPopupButton(agentName) {
    const button = $(`.agent-training-popup-btn[data-agent="${agentName}"]`).first();
    if (!button.length) {
        return;
    }
    button.prop('disabled', false)
        .attr('data-ready', 'true')
        .removeClass('bg-slate-200 text-slate-500 cursor-not-allowed')
        .addClass('bg-blue-600 text-white hover:bg-blue-700 cursor-pointer')
        .attr('title', 'Open training result summary');
}

function updateAgentEvaluationPopupButton(agentName) {
    const button = $(`.agent-evaluation-popup-btn[data-agent="${agentName}"]`).first();
    if (!button.length) {
        return;
    }
    button.prop('disabled', false)
        .attr('data-ready', 'true')
        .removeClass('bg-slate-200 text-slate-500 cursor-not-allowed')
        .addClass('bg-amber-500 text-white hover:bg-amber-600 cursor-pointer')
        .attr('title', 'Open evaluation result summary');
}

function handleAgentTrainingSummary(agentName, summary) {
    if (!window.lastAgentTrainingSummaries || typeof window.lastAgentTrainingSummaries !== 'object') {
        window.lastAgentTrainingSummaries = {};
    }
    window.lastAgentTrainingSummaries[agentName] = summary;
    updateAgentTrainingPopupButton(agentName);

    // Mark agent tab as finished
    setAgentMetricsTabStatus(agentName, 'finished');

    // Save to sessionStorage for recovery on page reload
    if (typeof saveTrainingSessionData === 'function') {
        saveTrainingSessionData();
    }

    showStatus(`Training summary ready for ${agentName}.`, 'success');
}

window.handleAgentTrainingSummary = handleAgentTrainingSummary;

function handleAgentEvaluationSummary(agentName, summary) {
    if (!window.lastAgentEvaluationSummaries || typeof window.lastAgentEvaluationSummaries !== 'object') {
        window.lastAgentEvaluationSummaries = {};
    }
    window.lastAgentEvaluationSummaries[agentName] = summary;
    updateAgentEvaluationPopupButton(agentName);

    // Save to sessionStorage for recovery on page reload
    if (typeof saveTrainingSessionData === 'function') {
        saveTrainingSessionData();
    }

    showStatus(`Evaluation summary ready for ${agentName}.`, 'success');
}

window.handleAgentEvaluationSummary = handleAgentEvaluationSummary;
window.restoreAgentsChartsFromRawData = restoreAgentsChartsFromRawData;

$(document).on('click', '.agent-training-popup-btn', function (event) {
    event.preventDefault();
    event.stopPropagation();

    const agentName = $(this).data('agent');
    const summary = window.lastAgentTrainingSummaries && agentName
        ? window.lastAgentTrainingSummaries[agentName]
        : null;

    if (!summary) {
        showStatus(`Training summary not ready yet for ${agentName || 'this agent'}.`, 'info');
        return;
    }

    openInfoPopupHtml(
        renderAgentTrainingSummaryPopup(summary),
        `Training Result - ${agentName}`,
        '/static/images/gif/training.gif'
    );
});

$(document).on('click', '.agent-evaluation-popup-btn', function (event) {
    event.preventDefault();
    event.stopPropagation();

    const agentName = $(this).data('agent');
    const summary = window.lastAgentEvaluationSummaries && agentName
        ? window.lastAgentEvaluationSummaries[agentName]
        : null;

    if (!summary) {
        showStatus(`Evaluation summary not ready yet for ${agentName || 'this agent'}.`, 'info');
        return;
    }

    openInfoPopupHtml(
        renderAgentEvaluationSummaryPopup(summary),
        `Evaluation Result - ${agentName}`,
        '/static/images/gif/test.gif'
    );
});

function getDataset(host, color) {
    return {
        label: host,
        data: [],
        borderColor: color,
        backgroundColor: addTransparency(color, 0.2), // Proper RGBA
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: color,
        tension: 0.1,
        fill: false, // Don't fill area under the line
    };
}

function getLiElement(agentKey, agent, isHost) {
    let ilElement = $(`<li data-host-agent='${agentKey}'></li>`);
    ilElement.addClass("flex items-center  bg-white p-1 rounded shadow mb-1 mr-1");
    html = getHtmlGauce(agentKey);
    html += "<details close class='ml-3'>";
    html += getHtmlSummaryAgent(agent, agentKey)
    if (isHost) {
        html += getHtmlDivTrafficHost();
    }
    else {
        html += getHtmlDivTrafficNetwork();
    }
    html += getHtmlCommTimeline(agentKey);
    html += `</details>`;
    ilElement.html(html);
    var gaugeContainer = ilElement.find(`#gauge-accuracy-${agentKey}`);;
    initializeGauceAccuracy(gaugeContainer);
    return ilElement;
}

function getHtmlSummaryAgent(agent, agentKey) {
    let html = `<summary class="font-bold cursor-pointer">`;
    if (agent != "")
        html += `<h1 class="ml-2 font-bold uppercase text-gray-800">${agent}</h1>`;
    html += getHtmlMetrics();
    html += getHtmlCommBadge(agentKey);
    html += `</summary> `;
    return html;
}

// ====================================================================
// COMMUNICATION VISIBILITY (marl_pz: alert family vs policy-coordination family)
// ====================================================================

function getHtmlCommBadge(agentKey) {
    // Hidden until we know the active comm strategy (see updateCommBadge()) —
    // stays hidden entirely for CommStrategy.NONE (no coordinator, nothing to show).
    return `<span class="comm-badge hidden ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold" data-comm-badge="${agentKey}"></span>`;
}

function getHtmlCommTimeline(agentKey) {
    return `<div class="comm-timeline-wrapper hidden mt-1" data-comm-timeline-wrapper="${agentKey}">
        <div class="comm-timeline-log max-h-32 overflow-y-auto text-[10px] font-mono bg-gray-50 border rounded p-1"
             data-comm-timeline="${agentKey}"></div>
    </div>`;
}

const _COMM_TIMELINE_MAX_ENTRIES = 200;
const _commBadgeRevertTimers = {};   // agentKey -> setTimeout id (policy-coordination pulse)
const _lastCommAlertValue = {};      // agentKey -> last rendered alert value (dedupe timeline)

function _commBadgeClassesAlert(value, strategy) {
    if (strategy === 'uaq') {
        if (value === 2) return 'bg-red-100 text-red-700';
        if (value === 1) return 'bg-amber-100 text-amber-700';
        return 'bg-gray-200 text-gray-600';
    }
    return value ? 'bg-red-100 text-red-700' : 'bg-gray-200 text-gray-600';
}

function _commBadgeTextAlert(value, strategy) {
    if (strategy === 'uaq') {
        if (value === 2) return 'Confident';
        if (value === 1) return 'Uncertain';
        return 'Normal';
    }
    return value ? 'Alert' : 'Normal';
}

function _commBadgeClassesSync() {
    return 'bg-blue-100 text-blue-700';
}

function _commBadgeTextSync(eventType) {
    if (eventType === 'fedavg_sync') return 'Synced';
    if (eventType === 'policy_copy') return 'Copied';
    if (eventType === 'experience_share') return 'Shared';
    return 'Synced';
}

/**
 * Shows/updates the per-host comm badge for agentKey (`${agentTeamName}_${hostOrCoordinator}`).
 * `payload` is either an alert-family snapshot ({family:'alert', hostAlerts, ...}) with `hostName`
 * telling us which entry to read, or a policy-coordination event ({family:'policy_coordination', eventType}).
 */
function updateCommBadge(agentKey, hostName, payload) {
    const badge = $(`span[data-comm-badge='${agentKey}']`);
    if (badge.length === 0) return;

    if (payload.family === 'alert') {
        const alert = (payload.hostAlerts || {})[hostName];
        if (!alert) return;
        badge.removeClass().addClass(
            `comm-badge ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${_commBadgeClassesAlert(alert.value, payload.strategy)}`
        );
        badge.text(_commBadgeTextAlert(alert.value, payload.strategy));
    } else if (payload.family === 'policy_coordination') {
        badge.removeClass().addClass(
            `comm-badge ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${_commBadgeClassesSync()}`
        );
        badge.text(_commBadgeTextSync(payload.eventType));
        // Transient pulse: policy-coordination has no continuous per-step state
        // to show (unlike alerts), so revert to idle a few seconds after the event.
        if (_commBadgeRevertTimers[agentKey]) {
            clearTimeout(_commBadgeRevertTimers[agentKey]);
        }
        _commBadgeRevertTimers[agentKey] = setTimeout(() => {
            badge.removeClass().addClass(
                'comm-badge ml-2 inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-gray-200 text-gray-600'
            );
            badge.text('Idle');
        }, 3000);
    }
}

/**
 * Appends one formatted line to the per-host communication timeline log.
 * Alert-family entries are deduped on value-change (avoids a firehose of
 * identical "normal" lines every step); policy-coordination entries are
 * per-event and always appended.
 */
function appendCommTimelineEntry(agentKey, hostName, payload) {
    const wrapper = $(`div[data-comm-timeline-wrapper='${agentKey}']`);
    const log = $(`div[data-comm-timeline='${agentKey}']`);
    if (log.length === 0) return;
    wrapper.removeClass('hidden');

    let line = null;
    if (payload.family === 'alert') {
        const alert = (payload.hostAlerts || {})[hostName];
        if (!alert) return;
        if (_lastCommAlertValue[agentKey] === alert.value) return; // dedupe
        _lastCommAlertValue[agentKey] = alert.value;
        line = `${hostName} -> coord: ${alert.label.toUpperCase()} (${alert.value})`;
    } else if (payload.family === 'policy_coordination') {
        const d = payload.detail || {};
        if (payload.eventType === 'fedavg_sync') {
            line = `[ep${payload.episode}] FedAvg sync: ${d.synced_count} agents averaged`;
        } else if (payload.eventType === 'policy_copy') {
            line = `[ep${payload.episode}] ${d.source}(best,r=${Number(d.best_reward).toFixed(1)}) -> copied policy to ${(d.targets || []).join(', ')}`;
        } else if (payload.eventType === 'experience_share') {
            line = `[ep${payload.episode}] ${d.source} shared ${d.shared_count} high-reward transitions -> ${(d.targets || []).join(', ')}`;
        } else {
            line = `[ep${payload.episode}] ${payload.eventType}`;
        }
    }
    if (!line) return;

    log.append(`<div>${$('<div>').text(line).html()}</div>`);
    if (log.children().length > _COMM_TIMELINE_MAX_ENTRIES) {
        log.children().first().remove();
    }
    log.scrollTop(log[0].scrollHeight);
}

function getHtmlDivTrafficHost() {
    let html = `<div>`;
    //html += ` Accuracy: <span class="accuracy mr-2 font-bold">N/A</span>`;
    html += ` <span class="text-gray-700">Pkt R/T:</span> <span class="packets_rx mr-2 font-bold">N/A</span> / <span class="packets_tx mr-2 font-bold">N/A</span>`;
    html += ` (<span class="var_packets_rx mr-2 font-bold">N/A</span> / <span class="var_packets_tx mr-2 font-bold">N/A</span>)`;
    html += ` <span class="text-gray-700">Byte R/T:</span> <span class="bytes_rx mr-2 font-bold">N/A</span> / <span class="bytes_tx mr-2 font-bold">N/A</span>`;
    html += ` (<span class="var_bytes_rx mr-2 font-bold">N/A</span> / <span class="var_bytes_tx mr-2 font-bold">N/A</span>)`;
    html += `</div>`;
    return html;
}


function getHtmlDivTrafficNetwork() {
    let html = `<div>`;
    if (isClassificationEnv()) {
        html += ` <span class="text-gray-700">Pkt:</span> <span class="packets mr-2 font-bold">N/A</span>`;
        html += ` <span class="text-gray-700"></span> <span class="bytes mr-2 font-bold">N/A</span>`;
    } else {
        html += ` <span class="text-gray-700">Pkt:</span> <span class="packets mr-2 font-bold">N/A</span>`;
        html += ` (<span class="var_packets mr-2 font-bold">N/A</span>)`;
        html += ` <span class="text-gray-700"></span> <span class="bytes mr-2 font-bold">N/A</span>`;
        html += ` (<span class="var_bytes mr-2 font-bold">N/A</span>)`;
    }
    html += `</div>`;
    return html;
}

function getHtmlGauce(agentKey) {
    html = `<div class="gauge-container" id="gauge-accuracy-${agentKey}">
            <svg class="gauge-svg" viewBox="0 0 200 120">
                <path class="gauge-bg" d="M 20 100 A 80 80 0 0 1 180 100"></path>
                <path class="gauge-fill" d="M 20 100 A 80 80 0 0 1 180 100"></path>
            </svg>
            <div class="gauge-text">0%</div>
            <div class="gauge-label">Accuratezza</div>
        </div>`;
    return html;
}

function getHtmlMetrics() {
    let html = `E/S <span class="episode font-bold text-black" title="Episode">0</span>/<span title="Step" class="step mr-2 font-bold text-black">0</span>`;
    html += ` Status: <span class="status mr-2 font-bold text-black">IDLE</span>`;
    html += ` Action: <span class="action_choosen mr-2 font-bold text-black">N/A</span>`;
    html += ` Reward:<span class="reward mr-2 font-bold text-black" >N/A</span>`;
    html += ` Guessed: <span class="correct_predictions mr-2 font-bold text-black" title="Correct Predictions">N/A</span>`;
    return html;
}

function initializeGauceAccuracy(container) {
    const path = container.find('.gauge-fill');
    const pathLength = path[0].getTotalLength();
    path.css({
        'stroke-dasharray': pathLength,
        'stroke-dashoffset': pathLength
    });
    container.data('pathLength', pathLength);
}

// To update the gauge accuracy
function updateGauceAccuracy(container, currentValue, maxValue) {
    const path = container.find('.gauge-fill');
    const textElement = container.find('.gauge-text');
    const pathLength = container.data('pathLength');
    const percentage = Math.min(Math.max((currentValue / maxValue) * 100, 0), 100);
    const offset = pathLength - (pathLength * percentage / 100);

    // Update the path
    path.css('stroke-dashoffset', offset);

    // Change color based on percentage
    let color;
    if (percentage < 33) {
        color = '#f44336'; // Red
    } else if (percentage < 66) {
        color = '#ff9800'; // Orange
    } else {
        color = '#4CAF50'; // Green
    }
    path.css('stroke', color);

    // Animate the text
    const startValue = parseFloat(textElement.text()) || 0;
    $({ value: startValue }).animate({ value: percentage }, {
        duration: 800,
        easing: 'swing',
        step: function () {
            textElement.text(Math.round(this.value) + '%');
        }
    });
}

function updateAgentStepDataTable(agent, stepData) {
    // Implementation for updating stats table for a specific agent
    var host = "";
    var agentKey = agent;
    if (isMultiAgentMode) {
        host = agentKey.split("_").pop();
    }
    else if ((isSingleAgentHostObservableEnv() && stepData.host !== undefined)) {
        host = stepData.host;
        agentKey = `${agent}_${host}`;
    }

    var listItem = $(`li[data-host-agent='${agentKey}']`);
    if (listItem.length > 0) {
        var gaugeContainer = listItem.find(`#gauge-accuracy-${agentKey}`);
        if (gaugeContainer.length > 0 && stepData.correctPredictions !== undefined && stepData.step !== undefined && stepData.step > 0) {
            updateGauceAccuracy(gaugeContainer, stepData.correctPredictions / stepData.step * 100, 100);
        }
        var episodeSpan = listItem.find(".episode").first();
        episodeSpan.html(stepData.episode);
        var stepSpan = listItem.find(".step").first();
        stepSpan.html(stepData.step);
        var statusSpan = listItem.find(".status").first();
        if (stepData.status.id !== undefined) {
            statusId = stepData.status.id;
            isHo = false;
            //if env id attack with observable hosts, stepData.status.id is an array
            if (Array.isArray(stepData.status.id)) {
                isHo = true;
                //if all 0 is normal
                let allNormal = stepData.status.id.every(id => id === 0);
                if (allNormal) {
                    statusId = 0;
                }
                else {
                    statusId = 1;
                }
            }

            html = getStatusActionIcon(statusId, host);

            if (!html)
                console.warn("Html status is empty for status id " + statusId + " and host " + host);
            else {
                statusSpan.html(html);
                // if (html.includes("secure.gif")) {

                // }
                // else if (html.includes("ping.gif") || html.includes("udp.gif") || html.includes("tcp.gif")) {

                // }
                // else 
                if (html.includes("cyberterrorism.gif")) {
                    listItem.addClass("bg-orange-200");
                    listItem.removeClass("bg-red-200");
                    listItem.removeClass("bg-white");
                }
                else if (html.includes("ddos.gif")) {
                    listItem.addClass("bg-red-200");
                    listItem.removeClass("bg-orange-200");
                    listItem.removeClass("bg-white");
                }
                else {
                    listItem.addClass("bg-white");
                    listItem.removeClass("bg-orange-200");
                    listItem.removeClass("bg-red-200");
                }
                if (html.includes("opacity-50")) {
                    listItem.addClass("opacity-50");
                }
                else {
                    listItem.removeClass("opacity-50");
                }

            }

        }
        var actionChoosenSpan = listItem.find(".action_choosen").first();
        if (stepData.action.choosen !== undefined) {
            choosen = stepData.action.choosen;
            if (isHo && stepData.action.choosen >= 0) {
                choosen = 1;
            }
            actionChoosenSpan.html(getStatusActionIcon(choosen, host));
        }
        // actionChoosenSpan.html(stepData.action.choosen !== undefined ? stepData.action.choosen : "N/A");
        var rewardSpan = listItem.find(".reward").first();
        rewardSpan.html(stepData.reward !== undefined ? stepData.reward : "N/A");
        if (stepData.action.isCorrect !== undefined) {
            if (stepData.action.isCorrect) {
                // actionChoosenSpan.addClass("text-green-600");
                actionChoosenSpan.removeClass("opacity-50");
                rewardSpan.addClass("text-green-600");
                rewardSpan.removeClass("text-red-600");
            }
            else {
                actionChoosenSpan.addClass("opacity-50");
                //actionChoosenSpan.removeClass("text-green-600");
                rewardSpan.addClass("text-red-600");
                rewardSpan.removeClass("text-green-600");
            }
        }

        var correctPredictionsSpan = listItem.find(".correct_predictions").first();
        correctPredictionsSpan.html(stepData.correctPredictions !== undefined ? stepData.correctPredictions : "N/A");
        if (isClassificationEnv()) {
            var packetsSpan = listItem.find(".packets").first();
            packetsSpan.html(stepData.receivedPackets + " / " + stepData.transmittedPackets);
            var bytesSpan = listItem.find(".bytes").first();
            bytesSpan.html(formatBytes(stepData.receivedBytes) + " / " + formatBytes(stepData.transmittedBytes));
        }
        else if (stepData.host === COORDINATOR || (!isMultiAgentMode && !isSingleAgentHostObservableEnv())) {
            var packetsSpan = listItem.find(".packets").first();
            packetsSpan.html(stepData.packets);
            var bytesSpan = listItem.find(".bytes").first();
            bytesSpan.html(formatBytes(stepData.bytes));
            updateDataPercentageSpan(listItem.find(".var_packets").first(), stepData.packetsPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_bytes").first(), stepData.bytesPercentageChange);
        }
        else {
            var packetsRxSpan = listItem.find(".packets_rx").first();
            packetsRxSpan.html(stepData.receivedPackets);
            var bytesRxSpan = listItem.find(".bytes_rx").first();
            bytesRxSpan.html(formatBytes(stepData.receivedBytes));
            var packetsTxSpan = listItem.find(".packets_tx").first();
            packetsTxSpan.html(stepData.transmittedPackets);
            var bytesTxSpan = listItem.find(".bytes_tx").first();
            bytesTxSpan.html(formatBytes(stepData.transmittedBytes));
            updateDataPercentageSpan(listItem.find(".var_packets_rx").first(), stepData.receivedPacketsPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_bytes_rx").first(), stepData.receivedBytesPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_packets_tx").first(), stepData.transmittedPacketsPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_bytes_tx").first(), stepData.transmittedBytesPercentageChange);
        }

    }
}

function addTransparency(hexColor, alpha = 0.2) {
    // Remove '#' if present
    hexColor = hexColor.replace('#', '');
    // Parse RGB values
    const r = parseInt(hexColor.substring(0, 2), 16);
    const g = parseInt(hexColor.substring(2, 4), 16);
    const b = parseInt(hexColor.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Helper function to get a consistent color for a given agent name
function getAgentColor(agentName) {
    // Simple hash function to generate a unique, consistent color
    let hash = 0;
    for (let i = 0; i < agentName.length; i++) {
        hash = agentName.charCodeAt(i) + ((hash << 5) - hash);
    }
    let color = '#';
    for (let i = 0; i < 3; i++) {
        const value = (hash >> (i * 8)) & 0xFF;
        color += ('00' + value.toString(16)).substr(-2);
    }
    return color;
}


function getStatusActionIcon(id, host) {
    if (isClassificationEnv()) {
        if (id == 0) {
            return "Sleeping";
        }
        if (id == 1) {
            return "<img src='/static/images/gif/ping.gif' alt='PING' title='PING' class='inline w-5 h-5'/>";
        }
        if (id == 2) {
            return "<img src='/static/images/gif/udp.gif' alt='UDP' title='UDP' class='inline w-5 h-5'/>";
        }
        if (id == 3) {
            return "<img src='/static/images/gif/tcp.gif' alt='TCP' title='TCP' class='inline w-5 h-5'/>";
        }
    }
    else {
        if (host === "" || host === COORDINATOR) {
            if (id == 0) {
                return "<img src='/static/images/gif/secure.gif' alt='Normal' title='Normal' class='inline w-5 h-5'/>";
            }
            if (id == 1) {
                return "<img src='/static/images/gif/ddos.gif' alt='Attack' title='Attack' class='inline w-5 h-5'/>";
            }
        }
        //marl and host specific
        if (id == 0) {
            return "<img src='/static/images/gif/secure.gif' alt='Normal' title='Normal' class='inline w-5 h-5'/>";
        }
        if (id == 1) {
            return "<img src='/static/images/gif/ddos.gif' alt='Incoming attack' title='Incoming attack' class='inline w-5 h-5'/>";
        }
        if (id == 3) {
            return "<img src='/static/images/gif/ddos.gif' alt='Incoming attack' title='Incoming attack' class='inline w-5 h-5 opacity-50'/>";
        }
        if (id == 2) {
            return "<img src='/static/images/gif/cyberterrorism.gif' alt='Attacking' title='Attacking' class='inline w-5 h-5'/>";
        }
        if (id == 4) {
            return "<img src='/static/images/gif/cyberterrorism.gif' alt='Attacking' title='Attacking' class='inline w-5 h-5 opacity-50'/>";
        }
    }
}

function updateDataPercentageSpan(span, percentage) {
    if (percentage !== undefined) {
        span.html(percentage.toFixed(2) + "%");
        if (percentage > 0) {
            span.removeClass("text-green-600");
            span.addClass("text-red-600");
        }
        else {
            span.addClass("text-green-600");
            span.removeClass("text-red-600");
        }
    }
}


/**
 * Updates the Accuracy Line Chart when a new episode result is received from an agent.
 * (Questa è la versione corretta che gestisce l'allineamento dei punti e la creazione dinamica dell'agente.)
 * @param {object} newData - The live data packet received from the socket.
 */
function updateLineChartAccuracy(lineChartAccuracy, newData, host) {
    //if (!newData || !newData.metrics || !newData.agent) return;

    if (host == "") {
        host = newData.agent; //"agent";
    }
    const metrics = newData.metrics;
    const episode = metrics.episode;

    // Convert accuracy (assumed 0-1) to percentage (0-100)
    const accuracy = (metrics.accuracy || 0) * 100;

    if (episode === undefined || accuracy === undefined) return;

    // 1. Find or Create the Dataset for this Agent
    let agentDataset = lineChartAccuracy.data.datasets.find(
        ds => ds.label === host //agentName
    );

    let isNewAgent = false;
    if (!agentDataset) {
        // New agent detected: create a new dataset
        const color = getAgentColor(host);
        agentDataset = {
            label: host,
            data: [],
            borderColor: color,
            backgroundColor: addTransparency(color, 0.2), // Proper RGBA
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: color,
            tension: 0.1,
            fill: false, // Don't fill area under the line
        };
        lineChartAccuracy.data.datasets.push(agentDataset);
        isNewAgent = true;
    }

    // 2. Manage Shared X-Axis (Labels = Episodes)
    const labels = lineChartAccuracy.data.labels;
    let episodeIndex = labels.indexOf(episode);
    if (lineChartAccuracy.data.labels.length === 0) {
        // First episode being added, open the details
        $(`[data-agent="${host}"]`).find('.agent-charts-section').find('details').prop('open', true);
    }
    // If episode is new to the chart, add it to the shared labels
    if (episodeIndex === -1) {
        labels.push(episode);
        // Sort labels to ensure episodes are chronological on the X-axis
        labels.sort((a, b) => a - b);
        episodeIndex = labels.indexOf(episode);

        // CRITICAL: When a new episode is added, all existing datasets 
        // need padding (null values) to align data points correctly.
        if (!isNewAgent) {
            lineChartAccuracy.data.datasets.forEach(ds => {
                // If the dataset is shorter than the labels array, it needs padding
                if (ds.data.length < labels.length - 1) {
                    // Fill gaps before the current episode with nulls for correct rendering
                    while (ds.data.length < labels.length - 1) {
                        ds.data.push(null);
                    }
                }
            });
        }
    }

    // 3. Update Data Point
    // Ensure the agent's data array is the correct length before setting the value
    while (agentDataset.data.length <= episodeIndex) {
        agentDataset.data.push(null); // Pad with nulls for episodes this agent hasn't reported yet
    }

    // Set the actual accuracy value at the correct, sorted index
    agentDataset.data[episodeIndex] = accuracy;

    lineChartAccuracy.update();
}



function updateLineChartReward(lineChartReward, newData, host) {
    //if (!newData || !newData.metrics || !newData.agent) return;
    if (host == "") {
        host = newData.agent; //"agent";
    }
    const metrics = newData.metrics;
    const episode = metrics.episode;
    const cumulativeReward = metrics.cumulativeReward;

    if (episode === undefined || cumulativeReward === undefined) return;

    // 1. Find or Create the Dataset for this Agent
    let agentDataset = lineChartReward.data.datasets.find(
        ds => ds.label === host //agentName
    );

    let isNewAgent = false;
    if (!agentDataset) {
        // New agent detected: create a new dataset
        const color = getAgentColor(host);
        agentDataset = {
            label: host,
            data: [],
            borderColor: color,
            backgroundColor: addTransparency(color, 0.2), // Proper RGBA
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: color,
            tension: 0.1,
            fill: false, // Don't fill area under the line
        };
        lineChartReward.data.datasets.push(agentDataset);
        isNewAgent = true;
    }

    // 2. Manage Shared X-Axis (Labels = Episodes)
    const labels = lineChartReward.data.labels;
    let episodeIndex = labels.indexOf(episode);
    // if (lineChartReward.data.labels.length === 0) {
    //     // First episode being added, open the details
    //     $(`[data-agent="${host}"]`).find('.agent-charts-section').find('details')[1].prop('open', true);
    // }
    // If episode is new to the chart, add it to the shared labels
    if (episodeIndex === -1) {
        labels.push(episode);
        // Sort labels to ensure episodes are chronological on the X-axis
        labels.sort((a, b) => a - b);
        episodeIndex = labels.indexOf(episode);

        // CRITICAL: When a new episode is added, all existing datasets 
        // need padding (null values) to align data points correctly.
        if (!isNewAgent) {
            lineChartReward.data.datasets.forEach(ds => {
                // If the dataset is shorter than the labels array, it needs padding
                if (ds.data.length < labels.length - 1) {
                    // Fill gaps before the current episode with nulls for correct rendering
                    while (ds.data.length < labels.length - 1) {
                        ds.data.push(null);
                    }
                }
            });
        }
    }

    // 3. Update Data Point
    // Ensure the agent's data array is the correct length before setting the value
    while (agentDataset.data.length <= episodeIndex) {
        agentDataset.data.push(null); // Pad with nulls for episodes this agent hasn't reported yet
    }

    // Set the actual accuracy value at the correct, sorted index
    agentDataset.data[episodeIndex] = cumulativeReward;

    lineChartReward.update();
}

