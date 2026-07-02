/**
 * Real-Time Chart Management Logic.
 * This script initializes and updates two Chart.js graphs (Line Chart and Bar Chart)
 * using live data received via WebSocket (from the updateData function called in app.js).
 * Chart.js must be loaded in index.html before this script.
 * * Note: The functions showCharts() and hideCharts() must be called from app.js 
 * when the training starts and stops, respectively.
 */


// Global variables for Chart.js instances
let allInitialized = false;
let lineChartPackets = null; // Initialize to null
let lineChartBytes = null; // Initialize to null
let barChartPackets = null;  // Initialize to null
let barChartBytes = null;  // Initialize to null
let agentsChartAccuracy = {}; // Initialize to null
let agentsChartReward = {}; // Initialize to null
let agentsDivData = {}; // Initialize to null
let chartMetricsSection = null;
let chartTrafficSection = null;
let chartSection = null;

let isMultiAgentMode = false; // Flag to indicate if multi-agent mode is active
let hosts = []; // Array to keep track of configured hosts
let agents = []; // Array to keep track of configured agents
let lastHostTasks = {}; // Object to keep track of last tasks per host
let lastAgentStatuses = {}; // Object to keep track of last statuses per agent
let lastAgentStepStats = {}; // Object to keep track of last step stats per agent
let lastAgentTrainingSummaries = {}; // Final popup data per agent
let lastAgentEvaluationSummaries = {}; // Final evaluation popup data per agent
let pendingChartRestore = false;

// Chart data storage for recovery on page reload
let chartDataRaw = {
    accuracy: {},  // {agentName: {labels: [...], datasets: [{label, data}, ...]}}
    reward: {}
};


// ===================================================================
// CHART DATA PERSISTENCE
// ===================================================================

function saveChartDataToSession() {
    try {
        sessionStorage.setItem('chartDataRaw', JSON.stringify(chartDataRaw));
    } catch (e) {
        console.warn('Could not save chart data to sessionStorage:', e);
    }
}



function repopulateChartsFromRawData() {
    if (typeof window.restoreAgentsChartsFromRawData === 'function') {
        return window.restoreAgentsChartsFromRawData(chartDataRaw);
    }
    return false;
}


function applyPendingChartRestore() {
    if (!pendingChartRestore) {
        return false;
    }
    if (repopulateChartsFromRawData()) {
        pendingChartRestore = false;
        return true;
    }
    return false;
}

/**
 * Normalizes a metric value coming from /get_training_status's
 * agent_chart_data into the canonical { labels, datasets: {key: [...]} }
 * shape — the same shape updateData() in this file already builds live and
 * persists to sessionStorage, so both restore paths converge on one format.
 *
 * - Flat array (single-agent scenarios): one dataset keyed by the agent name,
 *   index+1 treated as the episode number.
 * - Object (multi-host marl_pz/marl scenarios): { host: [v1, v2, ...] },
 *   each host trains independently so index+1 is that host's own episode
 *   number, not a shared one.
 */
function _toHostShapedSeries(agent, raw) {
    if (raw === null || raw === undefined) return null;
    if (Array.isArray(raw)) {
        if (raw.length === 0) return null;
        return {
            labels: raw.map((_, i) => i + 1),
            datasets: { [agent]: raw },
        };
    }
    if (typeof raw === 'object') {
        const maxLen = Math.max(0, ...Object.values(raw).map(a => Array.isArray(a) ? a.length : 0));
        if (maxLen === 0) return null;
        return {
            labels: Array.from({ length: maxLen }, (_, i) => i + 1),
            datasets: raw,
        };
    }
    return null;
}

function restoreChartDataFromSession(chartDataFromServer) {
    try {
        if (chartDataFromServer) {
            // chartDataFromServer is /get_training_status's agent_chart_data:
            // { agentName: { accuracy: [...] | {host: [...]}, reward: [...] | {host: [...]} } }
            const normalized = { accuracy: {}, reward: {} };
            Object.keys(chartDataFromServer).forEach(agent => {
                const v = chartDataFromServer[agent] || {};
                const acc = _toHostShapedSeries(agent, v.accuracy);
                if (acc) normalized.accuracy[agent] = acc;
                const rew = _toHostShapedSeries(agent, v.reward);
                if (rew) normalized.reward[agent] = rew;
            });
            chartDataRaw = normalized;
        } else {
            const stored = sessionStorage.getItem('chartDataRaw');
            if (!stored) return false;
            chartDataRaw = JSON.parse(stored);
        }
        console.log('Restored chart data from session:', chartDataRaw);
        pendingChartRestore = true;
        applyPendingChartRestore();
        return true;
    } catch (e) {
        console.warn('Could not restore chart data from sessionStorage:', e);
        return false;
    }
}

// ===================================================================
//  CHART VISIBILITY AND INITIALIZATION CONTROLLERS (New/Modified)
// ===================================================================

/**
 * Shows the chart containers and initializes the charts.
 * This should be called when training starts.
 */

function showCharts() {
    // Show chart containers 
    chartSection.removeClass('hidden');
    chartTrafficSection.removeClass('hidden');

    // Initialize/Re-initialize charts
    initializeLineChartBytes();
    initializeLineChartPackets();
    initializeBarChartPackets();
    initializeBarChartBytes();

    //allInitialized = true;
}

/**
 * Hides the chart containers.
 * This should be called when training stops.
 */
function hideCharts() {
    // Hide chart containers
    chartSection.addClass("hidden");
    chartTrafficSection.addClass("hidden");
    chartMetricsSection.addClass("hidden");

    // Optionally destroy charts to free up memory (important for long-running sessions)
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
    datasetForAccuracy = [];
    datasetForReward = [];

    if (lineChartPackets) {
        lineChartPackets.destroy();
        lineChartPackets = null;
    }
    if (lineChartBytes) {
        lineChartBytes.destroy();
        lineChartBytes = null;
    }
    if (barChartPackets) {
        barChartPackets.destroy();
        barChartPackets = null;
    }
    if (barChartBytes) {
        barChartBytes.destroy();
        barChartBytes = null;
    }
}

function resetTrainingDashboardForNewRun() {
    hideCharts();

    allInitialized = false;
    pendingChartRestore = false;
    chartDataRaw = {
        accuracy: {},
        reward: {}
    };

    lastHostTasks = {};
    lastAgentStatuses = {};
    lastAgentStepStats = {};
    lastAgentTrainingSummaries = {};
    lastAgentEvaluationSummaries = {};

    window.lastHostTasks = lastHostTasks;
    window.lastAgentStatuses = lastAgentStatuses;
    window.lastAgentStepStats = lastAgentStepStats;
    window.lastAgentTrainingSummaries = lastAgentTrainingSummaries;
    window.lastAgentEvaluationSummaries = lastAgentEvaluationSummaries;

    sessionStorage.removeItem('trainingSessionData');
    sessionStorage.removeItem('chartDataRaw');

    if (typeof resetAgentMetricsTabState === 'function') {
        resetAgentMetricsTabState();
    }

    $('#websocket-log').empty();
    $('#host-task-data').empty();
    $('#real-time-status').empty();
    $('#packets').text('0');
    $('#bytes').text('0');
    $('#var_packets').text('0');
    $('#var_bytes').text('0');
    $('#trendIndicatorPackets').empty();
    $('#trendIndicatorBytes').empty();
}


// ===================================================================
//  MAIN UPDATE FUNCTION AND INITIALIZATION
// ===================================================================

/**
 * Main function called from app.js to update both charts.
 * This is triggered upon receiving a 'live_update' socket message with level 'data'.
 * @param {object} data - The data object received from the socket (InstantState.to_dict()).
 */
function updateData(data) {
    // Ensure the payload is a data packet before processing
    if (!data || data.level !== 'data') {
        console.warn("Update data called with invalid or non-'data' level payload.");
        return;
    }

    if (!allInitialized) {
        showCharts(); // Ensure charts are shown and initialized, when the browser has been opened again
        allInitialized = true;
        if (currentConfig.cfg && currentConfig.cfg.agents && currentConfig.cfg.hosts) {
            initializeAgentsCharts(currentConfig.cfg.isMultiAgent, currentConfig.cfg.agents, currentConfig.cfg.hosts); // Initialize all charts with config data        
        }
        setStatus(SYSTEM_STATUS.TRAINING_RUNNING, SYSTEM_STATUS.RUNNING);
        showStatus("System is running. Charts initialized.", "success");
    }

    if (data.trafficData) {
        if (data.trafficData.packets !== undefined && data.trafficData.packets !== null) {
            if (lineChartPackets) {
                updateLineChartPackets(data);
            }
        }
        if (data.trafficData.bytes !== undefined && data.trafficData.bytes !== null) {
            if (lineChartBytes) {
                updateLineChartBytes(data);
            }
        }
        if (data.trafficData.hostStatusesStructured) {
            lastAgentStatuses = data.trafficData.hostStatusesStructured;
            if (barChartBytes) {
                updateBarChartBytes(data);
            }
            if (barChartPackets) {
                realTimeStatus = updateBarChartPackets(data);
                updateStatusSpan(realTimeStatus);
            }
        }

        // marl_pz alert-family communication (per-step host<->coordinator messages).
        // This is env-level (not per top-level agent), so fan out to every
        // currently-configured agent's panel — each one shows the same host/
        // coordinator communication state in its own per-host <li> rows.
        const commData = data.trafficData.commData;
        if (commData && commData.family === 'alert' && currentConfig.cfg && currentConfig.cfg.agents) {
            currentConfig.cfg.agents.forEach(function (agentTeamName) {
                Object.keys(commData.hostAlerts || {}).forEach(function (hostName) {
                    const agentKey = `${agentTeamName}_${hostName}`;
                    if (typeof updateCommBadge === 'function') {
                        updateCommBadge(agentKey, hostName, commData);
                    }
                    if (typeof appendCommTimelineEntry === 'function') {
                        appendCommTimelineEntry(agentKey, hostName, commData);
                    }
                });
                if (commData.coordinatorBroadcast !== null && commData.coordinatorBroadcast !== undefined) {
                    const coordKey = `${agentTeamName}_${COORDINATOR}`;
                    if (typeof updateCommBadge === 'function') {
                        updateCommBadge(coordKey, COORDINATOR, {
                            family: 'alert',
                            strategy: commData.strategy,
                            hostAlerts: { [COORDINATOR]: { value: commData.coordinatorBroadcast, label: commData.coordinatorBroadcast ? 'alert' : 'normal' } },
                        });
                    }
                }
            });
        }
    }

    // marl_pz policy-coordination communication (discrete sync events — S2/S3/S4).
    // data.agent here IS meaningful: each event is emitted with agent_name=agent.name
    // (the top-level agent team currently training), unlike the env-level commData above.
    if (data.commEvent && data.agent) {
        const ev = data.commEvent;
        const targets = (ev.participants && ev.participants.length) ? ev.participants : ['ALL'];
        targets.forEach(function (hostOrAgentId) {
            const agentKey = `${data.agent}_${hostOrAgentId}`;
            if (typeof updateCommBadge === 'function') {
                updateCommBadge(agentKey, hostOrAgentId, ev);
            }
            if (typeof appendCommTimelineEntry === 'function') {
                appendCommTimelineEntry(agentKey, hostOrAgentId, ev);
            }
        });
    }

    if (data.stepData && data.agent !== null) {
        lastAgentStepStats[data.agent] = data.stepData;
        updateAgentStepDataTable(data.agent, data.stepData);
        // Save training session data when step stats update
        if (typeof saveTrainingSessionData === 'function') {
            saveTrainingSessionData();
        }
    }

    if (data.agentTrainingSummary && data.agent) {
        lastAgentTrainingSummaries[data.agent] = data.agentTrainingSummary;
        if (typeof window.handleAgentTrainingSummary === 'function') {
            window.handleAgentTrainingSummary(data.agent, data.agentTrainingSummary);
        }
    }

    if (data.agentEvaluationSummary && data.agent) {
        lastAgentEvaluationSummaries[data.agent] = data.agentEvaluationSummary;
        if (typeof window.handleAgentEvaluationSummary === 'function') {
            window.handleAgentEvaluationSummary(data.agent, data.agentEvaluationSummary);
        }
    }

    if (data.hostTasks) {
        lastHostTasks = data.hostTasks;
        updateHostTasksTable(lastHostTasks);
        // Save training session data when host tasks update
        if (typeof saveTrainingSessionData === 'function') {
            saveTrainingSessionData();
        }
    }

    // Check if charts are initialized before updating
    if (data.metrics && data.agent) {
        let host = "";
        if (isMultiAgentMode) {
            //algo_variant_host
            const parts = data.agent.split('_');
            if (parts.length > 1)
                host = parts[parts.length - 1]; // Extract host from agent name
            const agentTeamName = data.agent.replace('_' + host, ''); // Remove host part to get team name
            lineChartAccuracy = agentsChartAccuracy[agentTeamName];
            lineChartReward = agentsChartReward[agentTeamName];
        } else {
            lineChartAccuracy = agentsChartAccuracy[data.agent];
            lineChartReward = agentsChartReward[data.agent];
            host = "";
        }
        if (lineChartAccuracy) {
            updateLineChartAccuracy(lineChartAccuracy, data, host);
        }

        if (lineChartReward) {
            updateLineChartReward(lineChartReward, data, host);
        }

        // Store chart data for recovery on page reload
        try {
            const agentKey = isMultiAgentMode ? data.agent.replace('_' + host, '') : data.agent;
            // updateLineChartAccuracy/Reward substitute an empty host with the
            // agent name when matching/creating datasets — mirror that here so
            // the stored key matches what's actually rendered live.
            const datasetKey = host || data.agent;

            // Store accuracy data
            if (!chartDataRaw.accuracy[agentKey]) {
                chartDataRaw.accuracy[agentKey] = {
                    labels: [],
                    datasets: {}
                };
            }
            if (data.metrics.accuracy !== undefined) {
                chartDataRaw.accuracy[agentKey].labels.push(data.metrics.episode || 0);
                if (!chartDataRaw.accuracy[agentKey].datasets[datasetKey]) {
                    chartDataRaw.accuracy[agentKey].datasets[datasetKey] = [];
                }
                chartDataRaw.accuracy[agentKey].datasets[datasetKey].push(data.metrics.accuracy);
            }

            // Store reward data
            if (!chartDataRaw.reward[agentKey]) {
                chartDataRaw.reward[agentKey] = {
                    labels: [],
                    datasets: {}
                };
            }
            // Backend sends the field as `cumulativeReward`, not `reward`.
            if (data.metrics.cumulativeReward !== undefined) {
                chartDataRaw.reward[agentKey].labels.push(data.metrics.episode || 0);
                if (!chartDataRaw.reward[agentKey].datasets[datasetKey]) {
                    chartDataRaw.reward[agentKey].datasets[datasetKey] = [];
                }
                chartDataRaw.reward[agentKey].datasets[datasetKey].push(data.metrics.cumulativeReward);
            }

            saveChartDataToSession();

            // Notify tab manager that this agent has live chart data
            const tabAgentKey = isMultiAgentMode ? data.agent.replace('_' + host, '') : data.agent;
            if (typeof notifyAgentChartUpdated === 'function') {
                notifyAgentChartUpdated(tabAgentKey);
            }
        } catch (e) {
            console.warn('Could not store chart data:', e);
        }
    }

}

// Global scope functions to be called from app.js
window.showCharts = showCharts;
window.hideCharts = hideCharts;
window.resetTrainingDashboardForNewRun = resetTrainingDashboardForNewRun;
window.updateData = updateData;
window.applyPendingChartRestore = applyPendingChartRestore;
window.lastAgentTrainingSummaries = lastAgentTrainingSummaries;
window.lastAgentEvaluationSummaries = lastAgentEvaluationSummaries;


// Initial state: hide charts on document ready
$(document).ready(function () {
    chartMetricsSection = $('#charts-metrics-section');
    chartTrafficSection = $('#charts-traffic-section');
    chartSection = $('#charts-section');
    hideCharts();
});
