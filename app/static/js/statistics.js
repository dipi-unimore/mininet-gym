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

// Constant to define the maximum size of the rolling window for the line chart
const MAX_DATA_POINTS = 60;

function getTrendIcon(value) {
    // Converti l'input in numero intero
    const numericValue = parseInt(value, 10);

    if (isNaN(numericValue)) {
        return '<span class="text-gray-400 text-sm">NA</span>';
    }

    // Inizializza il contenuto della freccia (icona Lucide + classe Tailwind per il colore)
    let iconHtml = '';

    if (numericValue > 0) {
        // Tendenza positiva: Freccia in SU, Colore ROSSO (come per un aumento negativo nel mercato)
        // Se volessi il verde per l'aumento, cambierei 'text-red-500' in 'text-green-500' e viceversa
        iconHtml = `
            <span class="trend-icon text-red-500">
                <i data-lucide="arrow-up-right" class="w-6 h-6" style="display:inline"></i>
            </span>
        `;
    } else if (numericValue < 0) {
        // Tendenza negativa: Freccia in GIÙ, Colore VERDE
        iconHtml = `
            <span class="trend-icon text-green-500">
                <i data-lucide="arrow-down-right" class="w-6 h-6" style="display:inline"></i>
            </span>
        `;
    } else {
        // Nessuna tendenza: Nessun simbolo (o un trattino/punto se preferito)
        iconHtml = `<span class="text-gray-500 text-base">Nessun cambiamento</span>`;
    }

    return iconHtml;
}

function getClassificationEnvStatusIcon(trafficType) {
    let iconHtml = `
            <span class="status-icon  text-green-500">
                <i data-lucide="shield-check" class="w-6 h-6" style="display:inline"></i>${trafficType}
            </span>
        `;
    return iconHtml;
}



function getAttackEnvStatusIcon(isAttack) {
    let iconHtml = '';

    if (!isAttack) {
        iconHtml = `
            <span class="status-icon  text-green-500">
                <i data-lucide="shield-check" class="w-6 h-6" style="display:inline"></i>
            </span>
        `;
    } else {
        iconHtml = `
            <span class="status-icon  text-red-500">
                <i data-lucide="zap" class="w-6 h-6" style="display:inline"></i>
            </span>
        `;
    }

    return iconHtml;
}

function updateTrendIndicator(inputValue, trendIndicatorId) {
    const indicatorElement = document.getElementById(trendIndicatorId);
    const newContent = getTrendIcon(inputValue);
    indicatorElement.innerHTML = newContent;
    lucide.createIcons();
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



function addTransparency(hexColor, alpha = 0.2) {
    // Remove '#' if present
    hexColor = hexColor.replace('#', '');
    // Parse RGB values
    const r = parseInt(hexColor.substring(0, 2), 16);
    const g = parseInt(hexColor.substring(2, 4), 16);
    const b = parseInt(hexColor.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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
