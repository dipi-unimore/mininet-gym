/**
 * Initializes the line chart for displaying Accuracy per Episode.
 * @param {string[]} hosts - Array of host names to pre-initialize datasets. (Optional, if hosts are known upfront)
 */
function initializeLineChartAccuracy(agent, hostsDataset = [], canvasEl) {
    //if (lineChartAccuracy) lineChartAccuracy.destroy();
    canvasEl.parent().css("height", "400px");

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
                    text: agent + ' Accuracy by Host per Episode',
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
                    text: agent + ' Reward by host per Episode (Multi-Agent)',
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


function initializeAgentsCharts(isMultiAgent, configuredAgents, configuredHosts) {
    isMultiAgentMode = isMultiAgent;
    if (agents && agents.length > 0 && hosts && hosts.length > 0) {
        return;
    }
    if (!configuredAgents || configuredAgents.length === 0) {
        console.warn("No agents configured for chart initialization.");
        return;
    }
    if (!configuredHosts || configuredHosts.length === 0) {
        console.warn("No hosts configured for chart initialization.");
        return;
    }
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
        if (agents[i].toLowerCase().includes("supervised"))
            continue;
        // Clone the agent section template
        let agentSection = $("div.agent-section").first().clone();

        // Remove hidden class and set agent identifier
        agentSection.removeClass("hidden");
        agentSection.attr("data-agent", agents[i]);

        agentSection.find(".agent-title").text("Agent: " + agents[i]);
        $("#charts-metrics-section").append(agentSection);

        // Prepare datasets for this agent
        let datasetForAccuracyAgent = structuredClone(datasetForAccuracy);
        let datasetForRewardAgent = structuredClone(datasetForReward);

        // Get the canvas elements within this agent section
        let accuracyCanvas = agentSection.find("canvas.lineChartAccuracy").first();
        let rewardCanvas = agentSection.find("canvas.lineChartReward").first();

        // Initialize charts for this agent
        agentsChartAccuracy[agents[i]] = initializeLineChartAccuracy(agents[i], datasetForAccuracyAgent, accuracyCanvas);
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
        else {
            let agentKey = agents[i];
            let ilElement = getLiElement(agentKey, "", false);
            ulElement.append(ilElement);
        }
    }
}

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
    ilElement.addClass("flex items-center justify-between bg-white p-1 rounded shadow mb-1 mr-1");
    html = getHtmlGauce(agentKey);
    html += "<details close class='ml-2 text-sm'>";
    html += getHtmlSummaryAgent(agent)
    if (isHost) {
        html += getHtmlDivTrafficHost();
    }
    else {
        html += getHtmlDivTrafficNetwork();
    }
    html += `</details>`;
    ilElement.html(html);
    var gaugeContainer = ilElement.find(`#gauge-accuracy-${agentKey}`);;
    initializeGauceAccuracy(gaugeContainer);
    return ilElement;
}

function getHtmlSummaryAgent(agent) {
    let html = `<summary class="font-bold cursor-pointer">`;
    if (agent != "")
        html += `<h1 class="ml-2 font-bold uppercase text-gray-800">${agent}</h1>`;
    html += getHtmlMetrics();
    html += `</summary> `;
    return html;
}

function getHtmlDivTrafficHost() {
    let html = `<div>`;
    //html += ` Accuracy: <span class="accuracy mr-2 font-bold">N/A</span>`;
    html += ` <span class="text-gray-700">Pkt rec.:</span> <span class="packets_rx mr-2 font-bold">N/A</span>`;
    html += ` <span class="var_packets_rx mr-2 font-bold">N/A</span>`;
    html += ` <span class="text-gray-700">KB rec.:</span> <span class="bytes_rx mr-2 font-bold">N/A</span>`;
    html += ` <span class="var_bytes_rx mr-2 font-bold">N/A</span>`;
    html += ` <span class="text-gray-700">Pkt tr.:</span> <span class="packets_tx mr-2 font-bold">N/A</span>`;
    html += ` <span class="var_packets_tx mr-2 font-bold">N/A</span>`;
    html += ` <span class="text-gray-700">KB tr.:</span> <span class="bytes_tx mr-2 font-bold">N/A</span>`;
    html += ` <span class="var_bytes_tx mr-2 font-bold">N/A</span>`;
    html += `</div>`;
    return html;
}


function getHtmlDivTrafficNetwork() {
    let html = `<div>`;
    html += ` <span class="text-gray-700">Pkt:</span> <span class="packets mr-2 font-bold">N/A</span>`;
    html += ` <span class="var_packets mr-2 font-bold">N/A</span>`;
    html += ` <span class="text-gray-700">KB:</span> <span class="bytes mr-2 font-bold">N/A</span>`;
    html += ` <span class="var_bytes mr-2 font-bold">N/A</span>`;
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
    var agentKey = agent;
    var listItem = $(`li[data-host-agent='${agentKey}']`);
    var host = "";
    if (isMultiAgentMode) {
        host = agentKey.split("_").pop();
    }
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
            statusSpan.html(getStatusActionIcon(stepData.status.id, host));
        }
        var actionChoosenSpan = listItem.find(".action_choosen").first();
        if (stepData.action.choosen !== undefined) {
            actionChoosenSpan.html(getStatusActionIcon(stepData.action.choosen, host));
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
        if (stepData.packets) {
            var packetsSpan = listItem.find(".packets").first();
            packetsSpan.html(stepData.packets);
            var bytesSpan = listItem.find(".bytes").first();
            bytesSpan.html(stepData.bytes / 1000);
            updateDataPercentageSpan(listItem.find(".var_packets").first(), stepData.packetsPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_bytes").first(), stepData.bytesPercentageChange);
        }
        else {
            var packetsRxSpan = listItem.find(".packets_rx").first();
            packetsRxSpan.html(stepData.receivedPackets);
            var bytesRxSpan = listItem.find(".bytes_rx").first();
            bytesRxSpan.html(stepData.receivedBytes);
            var packetsTxSpan = listItem.find(".packets_tx").first();
            packetsTxSpan.html(stepData.transmittedPackets / 1000);
            var bytesTxSpan = listItem.find(".bytes_tx").first();
            bytesTxSpan.html(stepData.transmittedBytes / 1000);
            updateDataPercentageSpan(listItem.find(".var_packets_rx").first(), stepData.receivedPacketsPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_bytes_rx").first(), stepData.receivedBytesPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_packets_tx").first(), stepData.transmittedPacketsPercentageChange);
            updateDataPercentageSpan(listItem.find(".var_bytes_tx").first(), stepData.transmittedBytesPercentageChange);
        }

    }
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
        if (id == 2) {
            return "<img src='/static/images/gif/cyberterrorism.gif' alt='Attacking' title='Attacking' class='inline w-5 h-5'/>";
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
        host = "agent"; //newData.agent;
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
        host = "agent"; //newData.agent;
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



