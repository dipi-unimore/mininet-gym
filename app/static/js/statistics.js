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



function extractNumericSeries(rawValue, needPercentageConversion = false) {
    if (rawValue === null || rawValue === undefined || !Array.isArray(rawValue)) {
        return null;
    }

    if (needPercentageConversion) {
        return rawValue.map(value => Number(value) * 100); // Convert to percentage
    }
    return rawValue.map(value => Number(value));

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

function restoreChartDataFromSession(chartDataFromServer) {
    try {
        if (chartDataFromServer) {
            // Normalize several possible server payload shapes into chartDataRaw
            const payload = chartDataFromServer;
            // Case 1: already in desired shape: { accuracy: {...}, reward: {...} }
            if (payload.accuracy && payload.reward) {
                chartDataRaw = payload;
            } else if (payload.agent_chart_data) {
                // Case 2: payload contains agent_chart_data: { agent: {accuracy, reward} }
                const normalized = { accuracy: {}, reward: {} };
                Object.keys(payload.agent_chart_data).forEach(agent => {
                    const v = payload.agent_chart_data[agent] || {};
                    // accuracy: could be scalar, array, or nested under metrics
                    let acc = null;
                    if (v.accuracy !== undefined) acc = v.accuracy;
                    else if (v.metrics && v.metrics.accuracy !== undefined) acc = v.metrics.accuracy;
                    if (acc !== null) {
                        normalized.accuracy[agent] = Array.isArray(acc) ? acc : [acc];
                    }
                    // reward: could be array of numbers, or indicators array with cumulative_reward
                    let rew = null;
                    if (v.reward !== undefined) rew = v.reward;
                    else if (v.indicators && Array.isArray(v.indicators)) {
                        try { rew = v.indicators.map(i => i.cumulative_reward); } catch (e) { rew = null; }
                    } else if (v.metrics && v.metrics.reward !== undefined) rew = v.metrics.reward;
                    if (rew !== null) {
                        normalized.reward[agent] = Array.isArray(rew) ? rew : [rew];
                    }
                });
                chartDataRaw = normalized;
            } else {
                // Case 3: maybe payload is already a plain map { agent: {accuracy, reward} }
                const normalized = { accuracy: {}, reward: {} };
                Object.keys(payload).forEach(agent => {
                    const v = payload[agent] || {};
                    if (v === null) return;
                    if (v.accuracy !== undefined) normalized.accuracy[agent] = Array.isArray(v.accuracy) ? v.accuracy : [v.accuracy];
                    else if (v.metrics && v.metrics.accuracy !== undefined) normalized.accuracy[agent] = Array.isArray(v.metrics.accuracy) ? v.metrics.accuracy : [v.metrics.accuracy];
                    if (v.reward !== undefined) normalized.reward[agent] = Array.isArray(v.reward) ? v.reward : [v.reward];
                    else if (v.indicators && Array.isArray(v.indicators)) {
                        try { normalized.reward[agent] = v.indicators.map(i => i.cumulative_reward); } catch (e) { }
                    }
                });
                chartDataRaw = normalized;
            }
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

            // Store accuracy data
            if (!chartDataRaw.accuracy[agentKey]) {
                chartDataRaw.accuracy[agentKey] = {
                    labels: [],
                    datasets: {}
                };
            }
            if (data.metrics.accuracy !== undefined) {
                chartDataRaw.accuracy[agentKey].labels.push(data.metrics.episode || 0);
                if (!chartDataRaw.accuracy[agentKey].datasets[host]) {
                    chartDataRaw.accuracy[agentKey].datasets[host] = [];
                }
                chartDataRaw.accuracy[agentKey].datasets[host].push(data.metrics.accuracy);
            }

            // Store reward data
            if (!chartDataRaw.reward[agentKey]) {
                chartDataRaw.reward[agentKey] = {
                    labels: [],
                    datasets: {}
                };
            }
            if (data.metrics.reward !== undefined) {
                chartDataRaw.reward[agentKey].labels.push(data.metrics.episode || 0);
                if (!chartDataRaw.reward[agentKey].datasets[host]) {
                    chartDataRaw.reward[agentKey].datasets[host] = [];
                }
                chartDataRaw.reward[agentKey].datasets[host].push(data.metrics.reward);
            }

            saveChartDataToSession();
        } catch (e) {
            console.warn('Could not store chart data:', e);
        }
    }

}

// Global scope functions to be called from app.js
window.showCharts = showCharts;
window.hideCharts = hideCharts;
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
