// ===================================================================
//  LINE CHART: Total Traffic (Last 60 points)
// ===================================================================

/**
 * Initializes the line chart for total traffic (packets, bytes, and percentage changes).
 * This function should only be called once when training starts.
 */
function initializeLineChartPackets() {
    if (lineChartPackets) {
        lineChartPackets.destroy(); // Destroy existing chart if present
    }

    const ctx = document.getElementById('lineChartPackets');

    // Set a defined height for the container to prevent infinite scrolling issues
    // You should also ensure the parent HTML elements have proper height settings.
    const container = ctx.parentElement;
    if (container) {
        container.style.height = '400px';
    }

    // Initialize data with empty arrays
    const initialData = {
        labels: [],
        datasets: [
            {
                label: 'Packets',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                yAxisID: 'yPackets', // Absolute values
                fill: false
            },
            {
                label: 'Packets % Change',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                tension: 0.1,
                borderDash: [5, 5],
                yAxisID: 'yPercent', // Percentage change
                fill: false
            }

        ]
    };

    lineChartPackets = new Chart(ctx, {
        type: 'line',
        data: initialData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time Step'
                    }
                },
                yPackets: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Packets  (Absolute)'
                    },
                    grid: {
                        drawOnChartArea: true,
                    },
                    beginAtZero: true
                },
                yPercent: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Percentage Change (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    beginAtZero: true,
                    min: -100
                }
            },
            // plugins: {
            //     // title: {
            //     //     display: true,
            //     //     text: agent + ' Accuracy by Host per Episode (Multi-Agent)',
            //     //     font: {
            //     //         size: 16
            //     //     }
            //     // },
            //     // legend: {
            //     //     display: true
            //     // },
            //     tooltip: {
            //         callbacks: {
            //             label: function (context) {
            //                 let label = context.dataset.label || 'Agent';
            //                 if (label) {
            //                     label += ': ';
            //                 }
            //                 if (context.parsed.y !== null) {
            //                     label += context.parsed.y.toFixed(2) + '%';
            //                 }
            //                 return label;
            //             }
            //         }
            //     }
            // }
        }
    });
}

function initializeLineChartBytes() {
    if (lineChartBytes) {
        lineChartBytes.destroy(); // Destroy existing chart if present
    }

    const ctx = document.getElementById('lineChartBytes');

    // Set a defined height for the container to prevent infinite scrolling issues
    // You should also ensure the parent HTML elements have proper height settings.
    const container = ctx.parentElement;
    if (container) {
        container.style.height = '400px';
    }

    // Initialize data with empty arrays
    const initialData = {
        labels: [],
        datasets: [
            {
                label: 'Bytes',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                yAxisID: 'yBytes', // Share the same Y-axis for absolute values
                fill: false
            },
            {
                label: 'Bytes % Change',
                data: [],
                borderColor: 'rgb(255, 159, 64)',
                tension: 0.1,
                borderDash: [5, 5],
                yAxisID: 'yPercent', // Percentage change
                fill: false
            }
        ]
    };

    lineChartBytes = new Chart(ctx, {
        type: 'line',
        data: initialData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time Step'
                    }
                },
                yBytes: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Bytes (Absolute)'
                    },
                    grid: {
                        drawOnChartArea: true,
                    },
                    beginAtZero: true
                },
                yPercent: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Percentage Change (%)'
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    beginAtZero: true,
                    min: -100
                    // max: 100
                }
            }
        }
    });
}



/**
 * Updates the line chart with new data, implementing the rolling window logic.
 *
 * This function adds a new data point and ensures that:
 * 1. If nd.id is 1 (an event), the point marker is a red 'X' and larger.
 * 2. Otherwise, the point marker is a circle with the dataset's default color and standard size.
 * 3. The data and style arrays (including size, fill, and border color) are shifted to maintain MAX_DATA_POINTS.
 *
 * @param {object} newData - The live data packet received from the socket.
 */
function updateLineChartPackets(newData) {
    if (!lineChartPackets || !newData || !newData.trafficData) return;

    const newLabels = lineChartPackets.data.labels;
    const datasets = lineChartPackets.data.datasets;

    // Extract traffic data (packets, percentage, and ID for the condition)
    const nd = newData.trafficData;

    var packetsSpan = $("#packets");
    if (nd.packets !== undefined)
        packetsSpan.html(nd.packets);
    var varPacketsSpan = $("#var_packets");
    if (nd.packetsPercentageChange !== undefined) {
        varPacketsSpan.html(nd.packetsPercentageChange.toFixed(2) + "%");
        if (nd.packetsPercentageChange > 0) {
            varPacketsSpan.removeClass("text-green-600");
            varPacketsSpan.addClass("text-red-600");
        }
        else {
            varPacketsSpan.addClass("text-green-600");
            varPacketsSpan.removeClass("text-red-600");
        }
        updateTrendIndicator(nd.packetsPercentageChange, "trendIndicatorPackets");
    }

    // --- Configuration Constants for Points ---
    const eventColor = 'red';
    const defaultPointRadius = 3; // Standard circle size
    const eventPointRadius = 6;   // Larger size for the 'X' event marker


    // 1. Determine the new point style, color, and size based on nd.id
    const isEvent = nd.id === 1 && !isClassificationEnv();

    // Point style (shape): 'crossRot' for the event, 'circle' for normal.
    const newPointStyle = isEvent ? 'crossRot' : 'circle';

    // Determine the colors and size.
    // The default color is taken from the dataset's line color (borderColor).
    const defaultColor = datasets[0].borderColor || 'rgba(54, 162, 235, 1)';

    // The point color must be red for the event, and default otherwise.
    const newPointColor = isEvent ? eventColor : defaultColor;

    // The point radius must be larger for the event, and default otherwise.
    const newPointRadius = isEvent ? eventPointRadius : defaultPointRadius;


    // Determine the current step label
    let currentStep = newLabels.length + 1;
    if (newLabels.length >= MAX_DATA_POINTS) {
        // If window is full, we maintain the sequence by calculating the next step label
        currentStep = parseInt(newLabels[newLabels.length - 1] || 0) + 1;
    }


    // 2. Add the new label and data
    newLabels.push(currentStep);

    // Iterate over all datasets to add data, apply point style, color, and radius
    datasets.forEach((dataset, index) => {
        let newDataValue;

        // Assign the correct value to the dataset based on index
        if (index === 0) { // Example: Packets count
            newDataValue = nd.packets;
        } else if (index === 1) { // Example: Packets Percentage Change
            newDataValue = nd.packetsPercentageChange;
        }

        // Add the value
        dataset.data.push(newDataValue);


        // --- POINT STYLE MANAGEMENT (Shape) ---
        if (dataset.pointStyle && Array.isArray(dataset.pointStyle)) {
            dataset.pointStyle.push(newPointStyle);
        } else {
            // Initialize pointStyle array if it doesn't exist
            dataset.pointStyle = new Array(dataset.data.length - 1).fill('circle');
            dataset.pointStyle.push(newPointStyle);
        }

        // --- POINT RADIUS MANAGEMENT (Size) ---
        if (dataset.pointRadius && Array.isArray(dataset.pointRadius)) {
            dataset.pointRadius.push(newPointRadius);
        } else {
            // Initialize pointRadius array
            dataset.pointRadius = new Array(dataset.data.length - 1).fill(defaultPointRadius);
            dataset.pointRadius.push(newPointRadius);
        }


        // --- POINT FILL COLOR MANAGEMENT ---
        // Use pointBackgroundColor to control the fill color (especially for circles)
        if (dataset.pointBackgroundColor && Array.isArray(dataset.pointBackgroundColor)) {
            dataset.pointBackgroundColor.push(newPointColor);
        } else {
            // Initialize the array with the default color for existing data
            const initialDefaultColor = dataset.borderColor || 'rgba(54, 162, 235, 1)';
            dataset.pointBackgroundColor = new Array(dataset.data.length - 1).fill(initialDefaultColor);
            dataset.pointBackgroundColor.push(newPointColor);
        }

        // --- POINT BORDER COLOR MANAGEMENT (CRUCIAL for 'X' color) ---
        // Use pointBorderColor to control the border color (the actual 'X' color when using crossRot)
        if (dataset.pointBorderColor && Array.isArray(dataset.pointBorderColor)) {
            dataset.pointBorderColor.push(newPointColor);
        } else {
            // Initialize the array with the default color for existing data
            const initialDefaultColor = dataset.borderColor || 'rgba(54, 162, 235, 1)';
            dataset.pointBorderColor = new Array(dataset.data.length - 1).fill(initialDefaultColor);
            dataset.pointBorderColor.push(newPointColor);
        }

    });

    // 3. Rolling Window Implementation:
    // If data points exceed MAX_DATA_POINTS, remove the oldest one (the first)
    if (newLabels.length > MAX_DATA_POINTS) {
        newLabels.shift();
        datasets.forEach(dataset => {
            dataset.data.shift();

            // Shift the oldest style, radius, and colors!
            if (dataset.pointStyle && Array.isArray(dataset.pointStyle)) {
                dataset.pointStyle.shift();
            }
            if (dataset.pointRadius && Array.isArray(dataset.pointRadius)) {
                dataset.pointRadius.shift();
            }
            if (dataset.pointBackgroundColor && Array.isArray(dataset.pointBackgroundColor)) {
                dataset.pointBackgroundColor.shift();
            }
            if (dataset.pointBorderColor && Array.isArray(dataset.pointBorderColor)) {
                dataset.pointBorderColor.shift();
            }
        });
    }

    // Update the chart
    lineChartPackets.update();
}

/**
 * Updates the line chart with new data, implementing the rolling window logic.
 * @param {object} newData - The live data packet received from the socket.
 */
function updateLineChartBytes(newData) {
    if (!lineChartBytes || !newData || !newData.trafficData) return;

    const newLabels = lineChartBytes.data.labels;
    const datasets = lineChartBytes.data.datasets;

    // Extract traffic data (packets, percentage, and ID for the condition)
    const nd = newData.trafficData;

    var bytesSpan = $("#bytes");
    if (nd.bytes !== undefined)
        bytesSpan.html(nd.bytes);
    var varBytesSpan = $("#var_bytes");
    if (nd.bytesPercentageChange !== undefined) {
        varBytesSpan.html(nd.bytesPercentageChange.toFixed(2) + "%");
        if (nd.bytesPercentageChange > 0) {
            varBytesSpan.removeClass("text-green-600");
            varBytesSpan.addClass("text-red-600");
        }
        else {
            varBytesSpan.addClass("text-green-600");
            varBytesSpan.removeClass("text-red-600");
        }
        updateTrendIndicator(nd.bytesPercentageChange, "trendIndicatorBytes");
    }


    // --- Configuration Constants for Points ---
    const eventColor = 'blue';
    const defaultPointRadius = 3; // Standard circle size
    const eventPointRadius = 6;   // Larger size for the 'X' event marker


    // 1. Determine the new point style, color, and size based on nd.id
    const isEvent = nd.id === 1 && !isClassificationEnv();

    // Point style (shape): 'crossRot' for the event, 'circle' for normal.
    const newPointStyle = isEvent ? 'crossRot' : 'circle';

    // Determine the colors and size.
    // The default color is taken from the dataset's line color (borderColor).
    const defaultColor = datasets[0].borderColor || 'rgba(54, 162, 235, 1)';

    // The point color must be red for the event, and default otherwise.
    const newPointColor = isEvent ? eventColor : defaultColor;

    // The point radius must be larger for the event, and default otherwise.
    const newPointRadius = isEvent ? eventPointRadius : defaultPointRadius;


    // Determine the current step label
    let currentStep = newLabels.length + 1;
    if (newLabels.length >= MAX_DATA_POINTS) {
        // If window is full, we maintain the sequence by calculating the next step label
        currentStep = parseInt(newLabels[newLabels.length - 1] || 0) + 1;
    }


    // 2. Add the new label and data
    newLabels.push(currentStep);

    // Iterate over all datasets to add data, apply point style, color, and radius
    datasets.forEach((dataset, index) => {
        let newDataValue;

        // Assign the correct value to the dataset based on index
        if (index === 0) { // Example: Packets count
            newDataValue = nd.bytes;
        } else if (index === 1) { // Example: Packets Percentage Change
            newDataValue = nd.bytesPercentageChange;
        }

        // Add the value
        dataset.data.push(newDataValue);


        // --- POINT STYLE MANAGEMENT (Shape) ---
        if (dataset.pointStyle && Array.isArray(dataset.pointStyle)) {
            dataset.pointStyle.push(newPointStyle);
        } else {
            // Initialize pointStyle array if it doesn't exist
            dataset.pointStyle = new Array(dataset.data.length - 1).fill('circle');
            dataset.pointStyle.push(newPointStyle);
        }

        // --- POINT RADIUS MANAGEMENT (Size) ---
        if (dataset.pointRadius && Array.isArray(dataset.pointRadius)) {
            dataset.pointRadius.push(newPointRadius);
        } else {
            // Initialize pointRadius array
            dataset.pointRadius = new Array(dataset.data.length - 1).fill(defaultPointRadius);
            dataset.pointRadius.push(newPointRadius);
        }


        // --- POINT FILL COLOR MANAGEMENT ---
        // Use pointBackgroundColor to control the fill color (especially for circles)
        if (dataset.pointBackgroundColor && Array.isArray(dataset.pointBackgroundColor)) {
            dataset.pointBackgroundColor.push(newPointColor);
        } else {
            // Initialize the array with the default color for existing data
            const initialDefaultColor = dataset.borderColor || 'rgba(54, 162, 235, 1)';
            dataset.pointBackgroundColor = new Array(dataset.data.length - 1).fill(initialDefaultColor);
            dataset.pointBackgroundColor.push(newPointColor);
        }

        // Use pointBorderColor to control the border color (the actual 'X' color when using crossRot)
        if (dataset.pointBorderColor && Array.isArray(dataset.pointBorderColor)) {
            dataset.pointBorderColor.push(newPointColor);
        } else {
            // Initialize the array with the default color for existing data
            const initialDefaultColor = dataset.borderColor || 'rgba(54, 162, 235, 1)';
            dataset.pointBorderColor = new Array(dataset.data.length - 1).fill(initialDefaultColor);
            dataset.pointBorderColor.push(newPointColor);
        }

    });

    // 3. Rolling Window Implementation:
    // If data points exceed MAX_DATA_POINTS, remove the oldest one (the first)
    if (newLabels.length > MAX_DATA_POINTS) {
        newLabels.shift();
        datasets.forEach(dataset => {
            dataset.data.shift();

            // Shift the oldest style, radius, and colors!
            if (dataset.pointStyle && Array.isArray(dataset.pointStyle)) {
                dataset.pointStyle.shift();
            }
            if (dataset.pointRadius && Array.isArray(dataset.pointRadius)) {
                dataset.pointRadius.shift();
            }
            if (dataset.pointBackgroundColor && Array.isArray(dataset.pointBackgroundColor)) {
                dataset.pointBackgroundColor.shift();
            }
            if (dataset.pointBorderColor && Array.isArray(dataset.pointBorderColor)) {
                dataset.pointBorderColor.shift();
            }
        });
    }

    lineChartBytes.update();
}

// ===================================================================
//  BAR CHART: Agent Traffic Breakdown
// ===================================================================

/**
 * Initializes the bar chart for agent traffic breakdown.
 * This function should only be called once when training starts.
 */
function initializeBarChartPackets() {
    if (barChartPackets) {
        barChartPackets.destroy(); // Destroy existing chart if present
    }

    const ctx = document.getElementById('barChartPackets');

    // Set a defined height for the container to prevent infinite scrolling issues
    const container = ctx.parentElement;
    if (container) {
        container.style.height = '400px';
    }

    barChartPackets = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [], // Populated with Host IDs (h1, h2, etc.)
            datasets: [
                {
                    label: 'Received Packets',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.7)', // Light Teal
                    stack: 'packets' // Stacks these two
                },
                {
                    label: 'Transmitted Packets',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.7)', // Blue
                    stack: 'packets' // Stacks these two
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Agent Traffic Breakdown (Current Snapshot)'
                }
            },
            scales: {
                x: {
                    stacked: false, // Prevents all 4 bars from stacking on top of each other. 
                    // The stack property in datasets creates two groups: 'packets' and 'bytes'.
                    title: {
                        display: true,
                        text: 'Host ID'
                    }
                },
                y: {
                    stacked: false,
                    title: {
                        display: true,
                        text: 'Traffic Value'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

function initializeBarChartBytes() {
    if (barChartBytes) {
        barChartBytes.destroy(); // Destroy existing chart if present
    }

    const ctx = document.getElementById('barChartBytes');

    // Set a defined height for the container to prevent infinite scrolling issues
    const container = ctx.parentElement;
    if (container) {
        container.style.height = '400px';
    }

    barChartBytes = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [], // Populated with Host IDs (h1, h2, etc.)
            datasets: [
                {
                    label: 'Received Bytes',
                    data: [],
                    backgroundColor: 'rgba(255, 99, 132, 0.7)', // Red
                    stack: 'bytes' // Stacks these two
                },
                {
                    label: 'Transmitted Bytes',
                    data: [],
                    backgroundColor: 'rgba(255, 159, 64, 0.7)', // Orange
                    stack: 'bytes' // Stacks these two
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Agent Traffic Breakdown (Current Snapshot)'
                }
            },
            scales: {
                x: {
                    stacked: false, // Prevents all 4 bars from stacking on top of each other. 
                    // The stack property in datasets creates two groups: 'packets' and 'bytes'.
                    title: {
                        display: true,
                        text: 'Host ID'
                    }
                },
                y: {
                    stacked: false,
                    title: {
                        display: true,
                        text: 'Traffic Value'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Updates the bar chart with the latest agent traffic data.
 *
 * This function processes traffic data for multiple hosts (agents)
 * and highlights any host where status.id is 1 (indicating an event/attack)
 * by setting a red border around the corresponding bars.
 *
 * @param {object} newData - The live data packet received from the socket.
 */
function updateBarChartPackets(newData) {
    if (!barChartPackets || !newData || !newData.trafficData || !newData.trafficData.hostStatusesStructured) return;

    // Extract raw traffic data and structured host statuses
    const nd = newData.trafficData
    const agentStatuses = nd.hostStatusesStructured;
    if (!agentStatuses) return;

    const hostLabels = Object.keys(agentStatuses);
    const receivedPacketsData = [];
    const transmittedPacketsData = [];

    // Arrays to store conditional styling (Border Color) for each bar
    const receivedBorderColors = [];
    const transmittedBorderColors = [];

    // Define colors
    const defaultReceivedColor = 'rgba(75, 192, 192, 1)';   // Default border color for received
    const defaultTransmittedColor = 'rgba(255, 159, 64, 1)'; // Default border color for transmitted
    const eventIncomingHighlightColor = 'red';
    const eventOutcomingHighlightColor = 'blu';
    const eventBorderWidth = 3; // Wider border for emphasis
    const defaultBorderWidth = 1;

    // Prepare data and styling for all agents/hosts
    var isAttack = false;
    var attackingHost = "";
    var victimHost = "";
    var trafficType = "";
    hostLabels.forEach(hostId => {
        const status = agentStatuses[hostId];

        // 1. Collect Data
        receivedPacketsData.push(status.receivedPackets || 0);
        transmittedPacketsData.push(status.transmittedPackets || 0);

        // 2. Determine Styling based on status.id
        if (isClassificationEnv()) {
            receivedBorderColors.push(defaultReceivedColor);
            transmittedBorderColors.push(defaultTransmittedColor);
            if (status.id === 1)
                trafficType = "PING";
            else if (status.id === 2)
                trafficType = "UDP";
            else if (status.id === 3)
                trafficType = "TCP";
        }
        else {

            const isEventIncoming = status.id === 1;
            const isEventOutcoming = status.id === 2;

            // Received Packets Styling
            if (isEventIncoming) {
                victimHost = hostId;
                isAttack = true;
                receivedBorderColors.push(eventIncomingHighlightColor);
                transmittedBorderColors.push(eventIncomingHighlightColor);
            }
            else if (isEventOutcoming) {
                attackingHost = hostId;
                isAttack = true;
                receivedBorderColors.push(eventOutcomingHighlightColor);
                transmittedBorderColors.push(eventOutcomingHighlightColor);
            }
            else {
                // Use the dataset's default border color
                receivedBorderColors.push(defaultReceivedColor);
                transmittedBorderColors.push(defaultTransmittedColor);
            }
        }

    });


    lucide.createIcons();

    // Update host labels
    barChartPackets.data.labels = hostLabels;

    // --- Update Dataset 0 (Received Packets) ---
    const receivedDataset = barChartPackets.data.datasets[0];
    receivedDataset.data = receivedPacketsData;

    // Apply conditional border colors and width
    receivedDataset.borderColor = receivedBorderColors;
    // Set borderWidth as an array if needed for individual control, or globally if only the event bar changes
    // Since we only want to change width for the event bar, we need to pass an array of widths.
    receivedDataset.borderWidth = hostLabels.map(hostId =>
        agentStatuses[hostId].id === 1 ? eventBorderWidth : defaultBorderWidth
    );

    // --- Update Dataset 1 (Transmitted Packets) ---
    const transmittedDataset = barChartPackets.data.datasets[1];
    transmittedDataset.data = transmittedPacketsData;

    // Apply conditional border colors and width
    transmittedDataset.borderColor = transmittedBorderColors;
    transmittedDataset.borderWidth = hostLabels.map(hostId =>
        agentStatuses[hostId].id === 1 ? eventBorderWidth : defaultBorderWidth
    );

    // Update the chart to reflect the changes
    barChartPackets.update();
    realTimeStatus = {
        isAttack: isAttack,
        victimHost: victimHost,
        attackingHost: attackingHost,
        trafficType: trafficType
    };
    return realTimeStatus;
}

function updateBarChartBytes(newData) {
    if (!barChartBytes || !newData || !newData.trafficData || !newData.trafficData.hostStatusesStructured) return;

    // Extract raw traffic data and structured host statuses
    const nd = newData.trafficData
    const agentStatuses = nd.hostStatusesStructured;
    if (!agentStatuses) return;

    const hostLabels = Object.keys(agentStatuses);
    const receivedBytesData = [];
    const transmittedBytesData = [];

    // Arrays to store conditional styling (Border Color) for each bar
    const receivedBorderColors = [];
    const transmittedBorderColors = [];

    // Define colors
    const defaultReceivedColor = 'rgba(75, 192, 192, 1)';   // Default border color for received
    const defaultTransmittedColor = 'rgba(255, 159, 64, 1)'; // Default border color for transmitted
    const eventIncomingHighlightColor = 'red';
    const eventOutcomingHighlightColor = 'blu';
    const eventBorderWidth = 3; // Wider border for emphasis
    const defaultBorderWidth = 1;

    // Prepare data and styling for all agents/hosts
    hostLabels.forEach(hostId => {
        const status = agentStatuses[hostId];

        // 1. Collect Data
        receivedBytesData.push(status.receivedBytes || 0);
        transmittedBytesData.push(status.transmittedBytes || 0);

        if (isClassificationEnv()) {
            receivedBorderColors.push(defaultReceivedColor);
            transmittedBorderColors.push(defaultTransmittedColor);
        }
        else {

            // 2. Determine Styling based on status.id
            const isEventIncoming = status.id === 1;
            const isEventOutcoming = status.id === 2;

            // Received Packets Styling
            if (isEventIncoming) {
                receivedBorderColors.push(eventIncomingHighlightColor);
            }
            else if (isEventOutcoming) {
                receivedBorderColors.push(eventOutcomingHighlightColor);
            }
            else {
                // Use the dataset's default border color
                receivedBorderColors.push(defaultReceivedColor);
            }

            // Transmitted Packets Styling
            if (isEventOutcoming) {
                transmittedBorderColors.push(eventIncomingHighlightColor);
            }
            else if (isEventIncoming) {
                transmittedBorderColors.push(eventOutcomingHighlightColor);
            } else {
                // Use the dataset's default border color
                transmittedBorderColors.push(defaultTransmittedColor);
            }
        }

    });

    // Update host labels
    barChartBytes.data.labels = hostLabels;

    // --- Update Dataset 0 (Received Packets) ---
    const receivedDataset = barChartBytes.data.datasets[0];
    receivedDataset.data = receivedBytesData;

    // Apply conditional border colors and width
    receivedDataset.borderColor = receivedBorderColors;
    // Set borderWidth as an array if needed for individual control, or globally if only the event bar changes
    // Since we only want to change width for the event bar, we need to pass an array of widths.
    receivedDataset.borderWidth = hostLabels.map(hostId =>
        agentStatuses[hostId].id === 1 ? eventBorderWidth : defaultBorderWidth
    );

    // --- Update Dataset 1 (Transmitted Packets) ---
    const transmittedDataset = barChartBytes.data.datasets[1];
    transmittedDataset.data = transmittedBytesData;

    // Apply conditional border colors and width
    transmittedDataset.borderColor = transmittedBorderColors;
    transmittedDataset.borderWidth = hostLabels.map(hostId =>
        agentStatuses[hostId].id === 1 ? eventBorderWidth : defaultBorderWidth
    );

    barChartBytes.update();
}

function updateStatusSpan(realTimeStatus) {
    if (!realTimeStatus) return;
    var statusSpan = $("#real-time-status");
    if (isClassificationEnv()) {
        var statusIcon = getClassificationEnvStatusIcon(realTimeStatus.trafficType);
        statusSpan.html(statusIcon);
        statusSpan.addClass("text-green-600");
        return;
    };
    isAttack = realTimeStatus.isAttack
    var statusIcon = getAttackEnvStatusIcon(isAttack);
    if (isAttack) {
        victimHost = realTimeStatus.victimHost;
        attackingHost = realTimeStatus.attackingHost;
        statusSpan.html("<img src='/static/images/gif/cyberterrorism.gif' alt='Attack' title='Attack' class='inline-block w-11 h-11'>" + attackingHost + statusIcon + "<img src='/static/images/gif/bombs.gif' alt='Attack' title='Attack' class='inline-block w-11 h-11'>" + victimHost);
        statusSpan.removeClass("text-green-600");
        statusSpan.addClass("text-red-600");
    }
    else {
        statusSpan.html(statusIcon);
        statusSpan.addClass("text-green-600");
        statusSpan.removeClass("text-red-600");
    }
}


function updateHostTasksTable(hostTasks) {
    var keys = Object.keys(hostTasks)
    if (keys.length > 0) {
        var hostTaskData = $("#host-task-data");
        hostTaskData.html("");
        keys.forEach(function (host) {
            var task = hostTasks[host];
            isLinkOff = task.linkStatus === 0;
            dstHost = task.destination;
            taskType = task.taskType;
            if (taskType === "normal") {
                trafficType = task.trafficType;
            } else {
                trafficType = "dos";
            }
            var taskItem = $("<li></li>");
            taskItem.addClass("flex items-center justify-between bg-white p-1 rounded shadow mb-1 mr-1");
            trafficTypeClass = "text-blue-600";
            imgLock = "";
            dstHostClass = "";
            dstImgClass = "";
            hidden = "";
            if (isLinkOff) {
                imgSrcFile = "host-locked";
                trafficTypeClass = "text-gray-600 line-through opacity-30";
                imgDstFile = "secure";
                imgLock = `<img src='/static/images/gif/firewall.gif' class='inline-block w-10 h-10'>`;
                dstHostClass = "opacity-30";
                dstImgClass = "opacity-30";
            } else if (taskType === "normal") {
                imgSrcFile = "server-security";
                if (trafficType === "tcp") {
                    trafficTypeClass = "text-orange-600";
                    imgDstFile = "server";
                } else if (trafficType === "udp") {
                    imgDstFile = "speed";
                    trafficTypeClass = "text-yellow-600";
                } else {
                    imgDstFile = "secure";
                    trafficTypeClass = "text-green-600";
                    if (trafficType === "none") {
                        hidden = "hidden";
                        dstHost = "";
                        trafficType = " sleeping";
                        //imgSrcFile = "server-security";
                    }
                }
            } else {
                imgSrcFile = "cyberterrorism";
                imgDstFile = "ddos";
                taskItem.addClass("bg-red-100");
                trafficTypeClass = "font-bold blink_me text-brown-600";
            }
            imgSrcTitle = host;
            imgDstTitle = dstHost;
            imgSrc = `<img src='/static/images/gif/${imgSrcFile}.gif' alt='${imgSrcTitle}' title='${imgSrcTitle}' class='inline-block w-10 h-10'>`;
            imgDst = `<img src='/static/images/gif/${imgDstFile}.gif' alt='${imgDstTitle}' title='${imgDstTitle}' class='inline-block w-10 h-10 ${hidden} ${dstImgClass}'>`;
            taskItem.html(`<span class="ml-2 font-bold">${host}</span> ${imgSrc}${imgLock} <span class="w-16 text-center ${trafficTypeClass} ${hidden}">${trafficType.toUpperCase()}</span> ${imgDst} <span class="mr-2 font-bold ${hidden} ${dstHostClass}">${dstHost}</span>`);
            hostTaskData.append(taskItem);
        });
    }
}






