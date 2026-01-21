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
    }

    if (data.hostTasks) {
        lastHostTasks = data.hostTasks;
        updateHostTasksTable(lastHostTasks);
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
    }

}

// Global scope functions to be called from app.js
window.showCharts = showCharts;
window.hideCharts = hideCharts;
window.updateData = updateData;


// Initial state: hide charts on document ready
$(document).ready(function () {
    chartMetricsSection = $('#charts-metrics-section');
    chartTrafficSection = $('#charts-traffic-section');
    chartSection = $('#charts-section');
    hideCharts();
});
