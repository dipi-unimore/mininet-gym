// ====================================================================\
// UI HELPERS
// ====================================================================\

function showStatus(message, type = 'info') {
    const statusEl = $('#status-message');
    let colorClass;
    let iconSvg = '';

    switch (type) {
        case 'success':
            colorClass = 'bg-green-100 border-l-4 border-green-500 text-green-700';
            iconSvg = '<svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
            break;
        case 'error':
            colorClass = 'bg-red-100 border-l-4 border-red-500 text-red-700';
            iconSvg = '<svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
            break;
        case 'info':
        default:
            colorClass = 'bg-blue-100 border-l-4 border-blue-500 text-blue-700';
            iconSvg = '<svg class="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>';
            break;
    }

    statusEl.removeClass().addClass('fixed bottom-2 left-2 p-2 rounded-lg shadow-2xl font-medium transition duration-500 ease-in-out z-20 flex items-center ' + colorClass).html(iconSvg + message).fadeIn();

    clearTimeout(statusEl.data('timeout'));
    statusEl.data('timeout', setTimeout(() => statusEl.fadeOut(500), 5000));
}


function setStatus(newSystemStatus, message) {
    var img1 = $('#training-status-text span img.system-status-image1');
    img1.on('click', function () { })
    var oldMessage = img1.attr('alt') || "";
    if (newSystemStatus < SYSTEM_STATUS.IDLE || // Invalid/error status/disconnected
        (newSystemStatus == SYSTEM_STATUS.IDLE && systemStatus > SYSTEM_STATUS.IDLE) || // Prevent overwriting with IDLE, after reconnecting
        (newSystemStatus == systemStatus && message == oldMessage)) {  // No change      
        console.warn(message);
        return;
    }
    var img2 = $('#training-status-text span img.system-status-image2');

    if (newSystemStatus == SYSTEM_STATUS.EVALUATING_RUNNING) { // Evaluating
        $('#start-training-btn').addClass('hidden');
        $('#pause-training-btn').removeClass('hidden');
        $('#stop-training-btn').removeClass('hidden');
        //$('#training-status-text span').text(message);  
        img1.attr('src', '/static/images/gif/network.gif');
        img2.attr('src', '/static/images/gif/test.gif');
        img2.removeClass('hidden');
        img1.attr('alt', message);
        img1.attr('title', message);
        $('#training-status-text span').removeClass('blink_me');
    }
    else if (newSystemStatus == SYSTEM_STATUS.PLOTTING_TRAINING_DATA) { // Plotting
        $('#start-training-btn').addClass('hidden');
        $('#pause-training-btn').addClass('hidden');
        $('#stop-training-btn').addClass('hidden');
        //$('#training-status-text span').text(message); 
        img1.attr('src', '/static/images/gif/evolution.gif');
        img2.addClass('hidden');
        img1.attr('alt', message);
        img1.attr('title', message);
        $('#training-status-text span').removeClass('blink_me');
    }
    else if (newSystemStatus == SYSTEM_STATUS.TRAINING_STARTING) { //starting 
        $('#start-training-btn').addClass('hidden');
        $('#start-training-btn-text').text("RESUME")
        $('#pause-training-btn').removeClass('hidden');
        $('#stop-training-btn').removeClass('hidden');
        img1.attr('src', '/static/images/gif/network.gif');
        img1.attr('alt', message);
        img1.attr('title', message);
        $('#training-status-text span').removeClass('blink_me');
    }
    else if (newSystemStatus == SYSTEM_STATUS.TRAINING_RUNNING) { // Running
        $('#start-training-btn').addClass('hidden');
        $('#pause-training-btn').removeClass('hidden');
        $('#stop-training-btn').removeClass('hidden');
        img1.attr('src', '/static/images/gif/network.gif');
        img2.attr('src', '/static/images/gif/training.gif');
        img2.removeClass('hidden');
        img1.attr('alt', message);
        img1.attr('title', message);
        $('#training-status-text span').removeClass('blink_me');
    }
    else if (newSystemStatus == SYSTEM_STATUS.PAUSED) { // Paused
        $('#start-training-btn').removeClass('hidden');
        $('#pause-training-btn').addClass('hidden');
        $('#stop-training-btn').removeClass('hidden');
        img1.attr('src', '/static/images/gif/pause.gif');
        img1.attr('alt', message);
        img1.attr('title', message);
        img2.addClass('hidden');
        $('#training-status-text span').addClass('blink_me');
    }
    else if (newSystemStatus == SYSTEM_STATUS.FINISHED) { // Finished
        $('#start-training-btn').removeClass('hidden');
        $('#start-training-btn-text').text("START TRAINING");
        $('#pause-training-btn').addClass('hidden');
        $('#stop-training-btn').addClass('hidden');
        img1.attr('src', '/static/images/gif/finish-line.gif');
        img1.on('click', function () {
            //open modal results
            $('#result-modal').removeClass('hidden');
        })
        img1.attr('alt', message);
        img1.attr('title', message);
        img2.addClass('hidden');
        $('#training-status-text span').removeClass('blink_me');
        //TODO: when finished, we can start again so we need to reset everything        
    }
    else if (newSystemStatus == SYSTEM_STATUS.RESUMED) { // Resume
        $('#start-training-btn').addClass('hidden');
        $('#pause-training-btn').removeClass('hidden');
        $('#stop-training-btn').removeClass('hidden');
        img1.attr('src', '/static/images/gif/network.gif');
        img1.attr('alt', message);
        img1.attr('title', message);
        img2.removeClass('hidden');
        $('#training-status-text span').removeClass('blink_me');
    }
    else {
        $('#start-training-btn').removeClass('hidden');
        $('#start-training-btn-text').text("START TRAINING");
        $('#pause-training-btn').addClass('hidden');
        $('#stop-training-btn').addClass('hidden');
        img1.attr('src', '/static/images/gif/start-line.gif');
        img1.attr('alt', message);
        img1.attr('title', message);
        img2.addClass('hidden');
        $('#training-status-text span').removeClass('blink_me');
    }
    systemStatus = newSystemStatus;
}

// ====================================================================\
// PAGE NAVIGATION
// ====================================================================\

function navigateTo(page) {
    currentPage = page;
    if (page === 'training') {
        // MANDATORY: Update config in memory on the backend before navigating
        updateConfigurationInMemory();

        $('#config-page').addClass('hidden');
        $('#training-page').removeClass('hidden');
        $('#nav-config').removeClass('bg-blue-600 hover:bg-blue-700 text-white').addClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
        $('#nav-training').addClass('bg-blue-600 hover:bg-blue-700 text-white').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');

        // Initialize WS only when on the training page
        initializeWebSocket();
    } else {
        $('#training-page').addClass('hidden');
        $('#config-page').removeClass('hidden');
        $('#nav-training').removeClass('bg-blue-600 hover:bg-blue-700 text-white').addClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
        $('#nav-config').addClass('bg-blue-600 hover:bg-blue-700 text-white').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');

        // Close WS if open
        if (socket && socket.connected) {
            socket.close();
        }
    }
}

// ====================================================================\
// CONFIGURATION RENDERING & HANDLING
// ====================================================================\

/**
 * Recursively generates a unique path for the input field.
 * @param {string} section - 'root', 'env_params', or 'agents'
 * @param {string} key - The current key (e.g., 'packets', 'ip')
 * @param {number|null} agentIndex - Index if dealing with agents
 * @param {string} nestedPath - Path from the section (e.g., 'net_params.controller')
 * @returns {string} Unique ID path (e.g., 'env_params.net_params.controller.ip')
 */
function buildInputId(section, key, agentIndex, nestedPath) {
    let path = section;
    if (agentIndex !== null) path += `-${agentIndex}`;
    if (nestedPath) path += `.${nestedPath}`;
    return `${path}.${key}`;
}

/**
 * Checks if a field is explicitly read-only.
 * @param {string} section 
 * @param {string} key 
 * @returns {boolean}
 */
function isFieldReadOnly(section, key) {
    return READ_ONLY_FIELDS.some(field => field.section === section && field.key === key);
}

/**
 * Creates an input field or select element HTML.
 * @param {string} key - The property name (e.g., 'packets')
 * @param {*} value - The current value
 * @param {string} fullId - The full path ID (e.g., 'env_params.controller.ip')
 * @param {boolean} isAgent - True if part of an agent config
 * @returns {string} HTML string
 */
function createInputField(key, value, fullId, isAgent) {
    const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const isReadonly = isFieldReadOnly(isAgent ? 'agents' : fullId.split('.').slice(0, -1).join('.'), key);

    let inputElement;
    let noteElement;

    // 1. Gym Type Select
    if (fullId === 'env_params.gym_type') {
        inputElement = `<select id="${fullId}" data-path="${fullId}" class="config-select">
                    ${GYM_TYPE_OPTIONS.map(opt =>
            `<option value="${opt}" ${value === opt ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
        noteElement = 'For \'from_dataset\' options, remember the dataset file. csv for classification, json for others'
    }
    // 2. Algorithm Select (Agents only)
    else if (key === 'algorithm' && isAgent) {
        inputElement = `<select id="${fullId}" data-path="${fullId}" class="config-select">
                    ${ALGORITHM_OPTIONS.map(opt =>
            `<option value="${opt}" ${value === opt ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
    }
    // 3. Log level Select
    else if (key === 'log_level') {
        inputElement = `<select id="${fullId}" data-path="${fullId}" class="config-select">
                    ${LOG_LEVELS.map(opt =>
            `<option value="${opt}" ${value === opt ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
    }
    // 4. Boolean Checkbox
    else if (typeof value === 'boolean') {
        inputElement = `<input type="checkbox" id="${fullId}" data-path="${fullId}" class="w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500" ${value ? 'checked' : ''} ${isReadonly ? 'disabled' : ''}>`;
    }
    // 5. Array (read-only text display)
    else if (Array.isArray(value)) {
        inputElement = `<input type="text" id="${fullId}" data-path="${fullId}" value="${value.join(', ')}" class="config-input" readonly>`;
    }
    // 6. Number or Text
    else {
        const type = typeof value === 'number' ? 'number' : 'text';
        let stepAttribute = '';

        if (type === 'number') {
            const valueString = value.toString();
            if (valueString.includes('.')) {
                // Calculate decimal places
                const decimalPart = valueString.split('.')[1] || '';
                const decimalPlaces = decimalPart.length;
                // Set step to 1 / (10^decimalPlaces), e.g., 0.001
                stepAttribute = `step="${Math.pow(10, -decimalPlaces).toFixed(decimalPlaces)}"`;
            } else {
                // Integer value, use step 1
                stepAttribute = 'step="1"';
            }
        }

        inputElement = `<input type="${type}" id="${fullId}" data-path="${fullId}" value="${value}" class="config-input" ${stepAttribute} ${isReadonly ? 'readonly' : ''}>`;

    }

    let fieldHtml = `
                <div class="${typeof value === 'boolean' ? 'flex items-center space-x-3' : 'space-y-1'}">
                    <label for="${fullId}" class="block text-sm font-medium text-gray-700 flex-shrink-0" title=" ${noteElement ?? ""}">${label} ${noteElement ? "(?)" : ""}</label>
                    ${inputElement}                   
                </div>
            `;
    return fieldHtml;
}

/**
 * Renders fields for a given configuration section, handling nesting.
 * @param {object} obj - The config object to render (e.g., currentConfig.env_params)
 * @param {string} path - The dotted path prefix (e.g., 'env_params')
 * @param {boolean} isAgent - Is this an agent?
 * @param {number|null} agentIndex - Agent index if applicable
 * @returns {string} HTML for the fields
 */
function renderFieldsRecursively(obj, path, isAgent = false, agentIndex = null) {
    let html = '';
    for (const key in obj) {
        if (!obj.hasOwnProperty(key)) continue;

        const value = obj[key];
        const currentPath = path ? `${path}.${key}` : key;
        const fullId = buildInputId(isAgent ? 'agents' : path.split('.')[0], key, agentIndex, path.split('.').slice(1).join('.'));

        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            // Start a new nested section container
            const sectionTitle = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            image = '';
            if (sectionTitle === "Net Params") {
                image = `<img src="/static/images/gif/network.gif" alt="Network params" title="Network params" class="inline-block w-6 h-6 ">`;
            }
            if (sectionTitle === "Attack Thresholds") {
                image = `<img src="/static/images/gif/ddos.gif" alt="Attack Thresholds" title="Attack Thresholds" class="inline-block w-6 h-6 ">`;
            }
            html += `
                        <div class="md:col-span-5 bg-gray-50 p-4 rounded-lg border-l-4 border-indigo-500">
                            <h5 class="text-md font-semibold text-gray-700 mb-3">${image}${sectionTitle}</h5>
                            <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
                                ${renderFieldsRecursively(value, currentPath, isAgent, agentIndex)}
                            </div>
                        </div>
                    `;
        } else {
            // Render the field
            html += createInputField(key, value, currentPath, isAgent);
        }
    }
    return html;
}

function renderConfig() {
    // 1. Render General (Root) Config
    const generalConfig = {};
    ROOT_KEYS.forEach(key => generalConfig[key] = currentConfig[key]);

    // Use 'root' as the path for the top-level items
    const generalHtml = renderFieldsRecursively(generalConfig, 'root');
    $('#general-config-list').html(generalHtml);

    // 2. Render Environment Parameters (env_params)
    const envHtml = renderFieldsRecursively(currentConfig.env_params, 'env_params');
    $('#env-params-config-list').html(envHtml);

    // 3. Render Agents
    renderAgents();
}

function renderAgents() {
    const agentsListEl = $('#agents-list');
    agentsListEl.empty();

    currentConfig.agents.forEach((agent, index) => {
        const agentHtml = `
                    <div id="agent-card-${index}" class="bg-gray-50 p-4 rounded-xl border border-gray-200 shadow-md">
                        <div class="flex justify-between items-start mb-3 border-b pb-2">
                            <h4 class="text-lg font-semibold text-gray-800">${agent.name}</h4>
                            <button data-agent-name="${agent.name}" class="remove-agent-btn text-red-500 hover:text-red-700 transition duration-150">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                            </button>
                        </div>
                        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-6 gap-4">
                            ${renderFieldsRecursively(agent, `agents.${index}`, true, index)}
                        </div>
                    </div>
                `;
        agentsListEl.append(agentHtml);
    });
}

/**
 * Finds and sets a value inside the config object using a dotted path.
 * Handles conversions to numbers/booleans.
 * @param {object} obj - The config object
 * @param {string} path - The dotted path (e.g., 'env_params.net_params.controller.ip')
 * @param {*} value - The new value string
 */
function setConfigValue(obj, path, value) {
    if (!path) {
        //console.info(obj);
        //console.info(value);                
        return;
    }
    const parts = path.split('.');
    let current = obj;

    for (let i = 0; i < parts.length - 1; i++) {
        if (!current[parts[i]]) current[parts[i]] = {};
        current = current[parts[i]];
    }

    const key = parts[parts.length - 1];
    let finalValue = value;

    // Type conversion
    if (value === 'true') finalValue = true;
    else if (value === 'false') finalValue = false;
    else if (!isNaN(parseFloat(value)) && isFinite(value)) finalValue = parseFloat(value);

    current[key] = finalValue;
}

function collectConfigFromUI() {
    let newConfig = JSON.parse(JSON.stringify(currentConfig)); // Deep copy

    // Collect all inputs and update the config based on the data-path attribute
    $('input, select').each(function () {
        const $input = $(this);
        const path = $input.data('path');
        let value;

        if ($input.attr('type') === 'checkbox') {
            value = $input.prop('checked');
        } else if ($input.prop('tagName') === 'SELECT') {
            value = $input.val();
        } else {
            value = $input.val();
        }

        // Only update if the field is not read-only
        if (!$input.prop('readonly') && !$input.prop('disabled')) {
            setConfigValue(newConfig, path, value);
        }
    });

    return newConfig;
}

// ====================================================================\
// AGENT MANAGEMENT
// ====================================================================\

function addAgent() {
    const newAgent = {
        name: "New_Agent_" + currentConfig.agents.length,
        algorithm: "DQN",
        enabled: true,
        episodes: 100,
        net_arch: [8, 8],
        learning_rate: 0.001,
        gamma: 0.99,
        show_action: false,
        load: false,
        save: true,
        progress_bar: true,
    };
    currentConfig.agents.push(newAgent);
    renderAgents();
    showStatus(`Agent ${newAgent.name} added.`, 'info');
}

function removeAgent(agentName) {
    const initialLength = currentConfig.agents.length;
    currentConfig.agents = currentConfig.agents.filter(agent => agent.name !== agentName);
    if (currentConfig.agents.length < initialLength) {
        renderAgents();
        showStatus(`Agent ${agentName} removed.`, 'success');
    }
}


function readYamlFile() {
    const fileInput = document.getElementById('yamlFileInput');
    const file = fileInput.files[0];

    if (!file) {
        alert('Select a YAML file first!');
        return;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
        try {
            const yamlContent = event.target.result;
            currentConfig = jsyaml.load(yamlContent); // Convert YAML to JS object
            // document.getElementById('yamlOutput').textContent = JSON.stringify(parsedData, null, 2);
            // console.log('YAML Data:', parsedData);
        } catch (error) {
            alert('Error parsing YAML file: ' + error.message);
        }
    };
    reader.readAsText(file);
    renderConfig();
    showStatus('YAML configuration loaded successfully.', 'success');
    updateConfigurationInMemory();
    $('#config-list-modal').addClass('hidden');
}

function closeResultModal() {
    $('#result-modal').addClass('hidden');
}

function renderConfigSummary() {
    const envEl = $('#config-env');
    envEl.empty();
    const preElement = $('<pre  class="max-h-[60vh] overflow-y-auto"></pre>').text(JSON.stringify(currentConfig.env_params, null, 2)).addClass('whitespace-pre-wrap bg-gray-100 p-4 rounded-lg text-sm text-gray-800');
    envEl.append('<h4 class="text-lg font-semibold text-gray-800 mb-2"><img src="static/images/gif/earth.gif" alt="Environment Parameters" title="Environment Parameters" class="inline-block w-6 h-6 ml-2">Environment Parameters</h4>');
    envEl.append(preElement);
    const agentsEl = $('#config-agents');
    agentsEl.empty();
    const preElementAgents = $('<pre  class="max-h-[60vh] overflow-y-auto"></pre>').text(JSON.stringify(currentConfig.agents, null, 2)).addClass('whitespace-pre-wrap bg-gray-100 p-4 rounded-lg text-sm text-gray-800');
    agentsEl.append('<h4 class="text-lg font-semibold text-gray-800 mb-2"><img src="static/images/gif/manager.gif" alt="Agents" title="Agents" class="inline-block w-6 h-6 ml-2">Agents</h4>');
    agentsEl.append(preElementAgents);
}




// ====================================================================\
// API INTERACTION
// ====================================================================\

// Function called when navigating from Config to Training page
function updateConfigurationInMemory() {
    const configToUpdate = collectConfigFromUI();
    currentConfig = configToUpdate; // Update local config after collecting

    $.ajax({
        url: '/update_config', // Backend API to update config in memory
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(configToUpdate),
        success: function (response) {
            showStatus(response.message || 'Configuration updated in memory successfully (no file save).', 'success');
        },
        error: function (xhr) {
            const response = xhr.responseJSON || { message: xhr.statusText };
            showStatus('Error updating config in memory: ' + response.message, 'error');
        }
    });
}

function downloadResults() {
    $.ajax({
        url: '/download_results', // Backend API to save to file and update memory
        type: 'POST',
        contentType: 'application/json',
        xhrFields: {
            responseType: 'blob' // Ricevi il file come blob
        },
        success: function (blob, status, xhr) {
            const downloadUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = xhr.getResponseHeader('Content-Disposition')
                .split('filename=')[1].replace(/"/g, '');
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(downloadUrl);

            showStatus('Results downloaded successfully.', 'success');
            closeResultModal();
        },
        error: function (xhr) {
            const response = xhr.responseJSON || { message: xhr.statusText };
            showStatus('Error downloading results: ' + response.message, 'error');
        }
    });
}

function saveConfiguration() {
    const configToSave = collectConfigFromUI();
    currentConfig = configToSave; // Update local config after collecting

    $.ajax({
        url: '/save_config', // Backend API to save to file and update memory
        type: 'POST',
        contentType: 'application/json',
        // headers: {
        //     Accept: 'application/octet-stream',
        // },
        data: JSON.stringify(configToSave),
        xhrFields: {
            responseType: 'blob' // Ricevi il file come blob
        },
        success: function (blob, status, xhr) {
            const downloadUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = xhr.getResponseHeader('Content-Disposition')
                .split('filename=')[1].replace(/"/g, '');
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(downloadUrl);

            showStatus('Configuration saved to file.', 'success');
        },
        error: function (xhr) {
            const response = xhr.responseJSON || { message: xhr.statusText };
            showStatus('Error saving configuration: ' + response.message, 'error');
        }
    });
}

// Other utilities
function formatBytes(num_bytes) {
    // 1. Manage value null and zero
    if (num_bytes === null || num_bytes === undefined || num_bytes === 0) {
        return "0";
    }

    // 2. Number must be positive
    const num = Math.abs(Number(num_bytes));

    const units = ["", "K", "M", "G", "T", "P"]; // P: Peta
    const base = 1000;

    // 3. Evaluate unit index
    const i = Math.floor(Math.log(num) / Math.log(base));
    const unitIndex = Math.min(units.length - 1, i);

    const unit = units[unitIndex];
    let value = num / Math.pow(base, unitIndex);
    let precision;

    // 4. Precision with 3 units
    if (unitIndex === 0) {
        // Value < 1000. No unit. No decimal.
        return Math.round(num).toString();
    }

    // Logica di precisione per mantenere ~3 cifre totali (XX.X, X.XX, XXX)
    if (value < 10) {
        // Es: 1.25K, 9.99K -> 2 decimali
        precision = 2;
    } else if (value < 100) {
        // Es: 20.3K, 99.9K -> 1 decimale
        precision = 1;
    } else {
        // Es: 203K, 999K -> 0 decimali
        precision = 0;
    }

    // 5. Formattazione e gestione del caso limite 999.x (overflow)
    let formattedValue = value.toFixed(precision);

    // Se l'arrotondamento causa l'overflow (es. 999.99K -> 1000.00K) e non siamo all'unità massima
    if (parseFloat(formattedValue) >= 1000 && unitIndex < units.length - 1) {
        return formatBytes(num * base);
    }

    // Remove 0 if present (eg. 20.0K -> 20K)
    return parseFloat(formattedValue).toString() + unit;
}


