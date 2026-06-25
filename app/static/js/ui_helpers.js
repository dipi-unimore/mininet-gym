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

const loadingOverlayReasons = new Set();

function ensureLoadingOverlay() {
    if (!$('#loading-overlay').length) {
        return;
    }
}

function updateLoadingOverlay() {
    ensureLoadingOverlay();
    const overlay = $('#loading-overlay');
    if (!overlay.length) {
        return;
    }

    if (loadingOverlayReasons.size > 0) {
        overlay.removeClass('hidden');
    } else {
        overlay.addClass('hidden');
        $('#loading-overlay-message').text('Loading...');
    }
}

function showLoadingOverlay(reason = 'global', message = 'Loading...') {
    ensureLoadingOverlay();
    loadingOverlayReasons.add(String(reason));
    $('#loading-overlay-message').text(message || 'Loading...');
    updateLoadingOverlay();
}

function hideLoadingOverlay(reason = 'global') {
    loadingOverlayReasons.delete(String(reason));
    updateLoadingOverlay();
}

function trackContainerImagesLoading(containerSelector, reason = 'images', message = 'Loading images...') {
    const container = $(containerSelector);
    if (!container.length) {
        return;
    }

    const images = container.find('img').toArray();
    if (!images.length) {
        return;
    }

    const loadingReason = String(reason);
    let remaining = images.length;
    showLoadingOverlay(loadingReason, message);

    const markDone = () => {
        remaining -= 1;
        if (remaining <= 0) {
            hideLoadingOverlay(loadingReason);
        }
    };

    images.forEach((img) => {
        if (img.complete) {
            markDone();
            return;
        }

        $(img).one('load.loading-spinner error.loading-spinner', markDone);
    });
}

function ensureInfoPopup(title = 'Info', icon = '/static/images/icon/info.png') {
    if ($('#info-popup-overlay').length) {
        $('#info-popup-title').text(title);
        $('#info-popup-icon').attr('src', icon);
        return;
    }

    const popupHtml = `
        <div id="info-popup-overlay" class="hidden fixed inset-0 bg-black bg-opacity-50 z-[120] flex items-center justify-center p-4">
            <div id="info-popup-card" class="bg-white rounded-xl shadow-2xl max-w-7xl w-[96vw] max-h-[92vh] flex flex-col border border-blue-100">
                <div class="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-blue-50 rounded-t-xl">
                    <div class="flex items-center gap-2">
                        <img id="info-popup-icon" src="${icon}" alt="Info" class="w-5 h-5">
                        <h4 id="info-popup-title" class="text-base font-semibold text-blue-800">${title}</h4>
                    </div>
                    <button id="info-popup-close" class="px-2 py-1 text-sm font-semibold text-gray-600 hover:text-gray-900" aria-label="Close">X</button>
                </div>
                <div class="px-4 py-4 flex-1 min-h-0 overflow-y-auto">
                    <div id="info-popup-text" class="text-sm text-gray-800 break-words"></div>
                </div>
                <div class="px-4 pb-4 flex justify-end">
                    <button id="info-popup-ok" class="px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium">OK</button>
                </div>
            </div>
        </div>
    `;

    $('body').append(popupHtml);
}

function closeInfoPopup() {
    $('#info-popup-overlay').addClass('hidden');
}

function openInfoPopup(message) {
    ensureInfoPopup();
    $('#info-popup-text').text(String(message || 'No additional information available.'));
    $('#info-popup-overlay').removeClass('hidden');
}

function openInfoPopupHtml(htmlContent, title = 'Info', icon = '/static/images/icon/info.png') {
    ensureInfoPopup(title, icon);
    $('#info-popup-text').html(String(htmlContent || '<p>No additional information available.</p>'));
    $('#info-popup-overlay').removeClass('hidden');
    trackContainerImagesLoading('#info-popup-text', 'info-popup-images', 'Loading preview images...');
}

$(document).on('click', 'img[src$="/static/images/icon/info.png"][title]', function (event) {
    event.preventDefault();
    event.stopPropagation();
    const titleText = $(this).attr('title') || '';
    openInfoPopup(titleText);
});

$(document).on('click', '#info-popup-close, #info-popup-ok', function () {
    closeInfoPopup();
});

$(document).on('click', '#info-popup-overlay', function (event) {
    if (event.target && event.target.id === 'info-popup-overlay') {
        closeInfoPopup();
    }
});

$(document).on('keydown', function (event) {
    if (event.key === 'Escape' && $('#info-popup-overlay').length && !$('#info-popup-overlay').hasClass('hidden')) {
        closeInfoPopup();
    }
});


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

function updateNavigationButtons(page) {
    const desktop = {
        config: $('#nav-config'),
        training: $('#nav-training'),
        results: $('#nav-results'),
    };
    const mobile = {
        config: $('#nav-config-mobile'),
        training: $('#nav-training-mobile'),
        results: $('#nav-results-mobile'),
    };

    const resetButton = (btn) => {
        btn.removeClass('bg-blue-600 hover:bg-blue-700 bg-green-600 hover:bg-green-700 text-white cy-nav-active')
            .addClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
    };

    Object.values(desktop).forEach(resetButton);
    Object.values(mobile).forEach(resetButton);

    if (page === 'results') {
        desktop.results.addClass('bg-green-600 hover:bg-green-700 text-white cy-nav-active').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
        mobile.results.addClass('bg-green-600 hover:bg-green-700 text-white cy-nav-active').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
    } else if (page === 'training') {
        desktop.training.addClass('bg-blue-600 hover:bg-blue-700 text-white cy-nav-active').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
        mobile.training.addClass('bg-blue-600 hover:bg-blue-700 text-white cy-nav-active').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
    } else {
        desktop.config.addClass('bg-blue-600 hover:bg-blue-700 text-white cy-nav-active').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
        mobile.config.addClass('bg-blue-600 hover:bg-blue-700 text-white cy-nav-active').removeClass('bg-gray-200 hover:bg-gray-300 text-gray-600');
    }

    $('#mobile-nav-toast').addClass('hidden');
    $('#mobile-nav-toggle').attr('aria-expanded', 'false');
}

async function navigateTo(page) {
    currentPage = page;
    if (page === 'training') {
        const synced = await syncTrainingParamsFromLoadedScenario();
        if (!synced) {
            return;
        }

        // MANDATORY: Update config in memory on the backend before navigating
        updateConfigurationInMemory();

        $('#config-page').addClass('hidden');
        $('#results-page').addClass('hidden');
        $('#training-page').removeClass('hidden');
        updateNavigationButtons('training');
        renderTrainingCaption();

        // Initialize WS only when on the training page
        initializeWebSocket();
    } else if (page === 'config') {
        $('#training-page').addClass('hidden');
        $('#results-page').addClass('hidden');
        $('#config-page').removeClass('hidden');
        updateNavigationButtons('config');

        // Close WS if open
        if (socket && socket.connected) {
            socket.close();
        }
    } else if (page === 'results') {
        $('#config-page').addClass('hidden');
        $('#training-page').addClass('hidden');
        $('#results-page').removeClass('hidden');
        updateNavigationButtons('results');
        renderResultsPanel();
    }
}

async function syncTrainingParamsFromLoadedScenario() {
    const envParams = (currentConfig && currentConfig.env_params) ? currentConfig.env_params : null;
    if (!envParams) return true;

    const source = String(envParams.scenario_source || '').toLowerCase();
    if (source !== 'load') {
        return true;
    }

    const selectedScenarioPath = String(envParams.scenario_file || '').trim();
    if (!selectedScenarioPath) {
        showStatus('Load existing is selected but no scenario file is set.', 'error');
        openInfoPopup('Select a scenario row before entering Training.');
        return false;
    }

    try {
        const response = await getScenarioDetails(selectedScenarioPath);
        const summary = (response && response.summary) ? response.summary : {};
        const statistics = (response && response.statistics) ? response.statistics : {};
        const training = statistics.training || {};
        const evaluation = statistics.evaluation || {};

        const toFiniteNumber = (value) => {
            const n = Number(value);
            return Number.isFinite(n) ? n : null;
        };

        const trainEpisodes = toFiniteNumber(summary.train_episodes ?? training.episodes);
        const trainMaxSteps = toFiniteNumber(summary.train_max_steps ?? training.max_steps);
        const evalEpisodes = toFiniteNumber(summary.eval_episodes ?? evaluation.episodes);

        if (trainEpisodes !== null) envParams.episodes = trainEpisodes;
        if (trainMaxSteps !== null) envParams.max_steps = trainMaxSteps;
        if (evalEpisodes !== null) envParams.test_episodes = evalEpisodes;

        const syncInputValue = (path, value) => {
            const input = $(`[data-path="${path}"]`);
            if (input.length && value !== null) {
                input.val(value);
            }
        };

        syncInputValue('env_params.episodes', trainEpisodes);
        syncInputValue('env_params.max_steps', trainMaxSteps);
        syncInputValue('env_params.test_episodes', evalEpisodes);

        showStatus('Training parameters synced from selected scenario.', 'success');
        return true;
    } catch (errorMessage) {
        const message = String(errorMessage || 'Unknown error');
        showStatus('Unable to sync training params from selected scenario: ' + message, 'error');
        openInfoPopup('Unable to load selected scenario details before Training.\n\n' + message);
        return false;
    }
}

function renderTrainingCaption() {
    const env = (currentConfig && currentConfig.env_params) ? currentConfig.env_params : {};
    const agents = Array.isArray(currentConfig.agents) ? currentConfig.agents : [];
    const net = env.net_params || {};
    const badges = [];

    // pre-built class strings so Tailwind CDN can detect them statically
    const B = {
        blue:   'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800',
        gray:   'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700',
        green:  'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800',
        amber:  'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800',
        indigo: 'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800',
    };
    const badge = (cls, text) => `<span class="${cls}" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;

    // Scenario
    if (env.gym_type) badges.push(badge(B.blue, env.gym_type));

    // Network topology
    const topoparts = [];
    if (net.num_switches) topoparts.push(`${net.num_switches} SW`);
    if (net.num_hosts)    topoparts.push(`${net.num_hosts} hosts`);
    if (net.num_iots)     topoparts.push(`${net.num_iots} IoT`);
    if (topoparts.length) badges.push(badge(B.gray, topoparts.join(' · ')));

    // Training params
    const isDatasetScenario = env.gym_type && env.gym_type.includes('_from_dataset');
    const trainparts = [];
    if (isDatasetScenario) {
        const selectedPath = env.data_traffic_file || '';
        const datasetItem = _datasetList.find(d => d.path === selectedPath);
        const entries = datasetItem ? ((datasetItem.summary && datasetItem.summary.entries) || 0) : null;
        const steps = env.max_steps || 0;
        const testEp = env.test_episodes || 0;
        if (entries !== null && steps > 0) {
            const computedEp = Math.floor((entries - testEp) / steps);
            trainparts.push(`${computedEp} ep`);
        }
    } else {
        if (env.episodes) trainparts.push(`${env.episodes} ep`);
    }
    if (env.max_steps)      trainparts.push(`${env.max_steps} steps`);
    if (env.test_episodes)  trainparts.push(`test: ${env.test_episodes} ep`);
    if (trainparts.length) badges.push(badge(B.green, trainparts.join(' · ')));

    // Dataset (only for _from_dataset scenarios)
    if (isDatasetScenario && env.data_traffic_file) {
        const fullPath = env.data_traffic_file;
        const MAX = 45;
        const display = fullPath.length > MAX ? '…' + fullPath.slice(-MAX) : fullPath;
        const datasetItem = _datasetList.find(d => d.path === fullPath);
        const entries = datasetItem ? ((datasetItem.summary && datasetItem.summary.entries) || 0) : null;
        const entriesSuffix = entries !== null ? ` · ${entries} entries` : '';
        badges.push(`<span class="${B.amber}" title="${escapeHtml(fullPath)}">${escapeHtml(display)}${escapeHtml(entriesSuffix)}</span>`);
    }

    // Agents
    agents.filter(a => a.enabled !== false).forEach(a => {
        const text = (a.name || 'Agent') + (a.algorithm ? ` (${a.algorithm})` : '');
        badges.push(badge(B.indigo, text));
    });

    $('#training-caption').html(badges.join(''));
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
 * Checks if a field is explicitly hidden.
 * @param {string} section 
 * @param {string} key 
 * @returns {boolean}
 */
function isFieldHidden(section, key) {
    return HIDDEN_FIELDS.some(field => field.section === section && field.key === key);
}


function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function resolveConfigComment(path) {
    if (!path || !window.configComments) return '';
    if (window.configComments[path]) return window.configComments[path];

    // Root section in UI uses 'root.<key>', YAML paths are '<key>'
    if (path.startsWith('root.')) {
        const rootPath = path.replace(/^root\./, '');
        if (window.configComments[rootPath]) return window.configComments[rootPath];
    }

    // Fallback for agent-indexed paths: agents.0.foo -> agents.*.foo
    const wildcardPath = path.replace(/^agents\.\d+\./, 'agents.*.');
    return window.configComments[wildcardPath] || '';
}

function loadConfigComments() {
    return new Promise((resolve) => {
        $.ajax({
            url: '/static/json/config_parameter_comments.json',
            type: 'GET',
            success: function (response) {
                window.configComments = response || {};
                resolve(window.configComments);
            },
            error: function () {
                window.configComments = {};
                resolve(window.configComments);
            }
        });
    });
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
    const isHidden = isFieldHidden(isAgent ? 'agents' : fullId.split('.').slice(0, -1).join('.'), key);
    const commentFromYaml = resolveConfigComment(fullId);

    let inputElement;
    let noteElement;

    // 1. Gym Type Select
    if (fullId === 'env_params.gym_type') {
        inputElement = `<select id="${fullId}" data-path="${fullId}" class="config-select">
                    ${GYM_TYPE_OPTIONS.map(opt =>
            `<option value="${opt}" ${value.toLowerCase() === opt.toLowerCase() ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
        noteElement = 'For \'from_dataset\' options, remember the json dataset file.'
        $(document).on('change', `#${CSS.escape(fullId)}`, async function () {
            currentConfig.env_params.gym_type = $(this).val();
            try {
                const scenarioParams = await getScenarioEnvParams(currentConfig.env_params.gym_type);
                if (scenarioParams && Object.keys(scenarioParams).length > 0) {
                    Object.assign(currentConfig.env_params, scenarioParams);
                    const sectionKey = Object.keys(scenarioParams)[0]; // 'attacks' or 'classification'
                    const sectionExists = $(`#${CSS.escape('env_params.' + sectionKey)}-section`).length > 0;
                    if (!sectionExists) {
                        // Cross-type switch: the target section was never rendered; re-render all env sub-sections.
                        if (!shouldShowScenarioSelector()) {
                            delete currentConfig.env_params.scenario_source;
                            delete currentConfig.env_params.scenario_file;
                        }
                        if (!shouldShowDatasetSelector()) {
                            delete currentConfig.env_params.data_traffic_file;
                        }
                        $(document).off('change', `#${CSS.escape(fullId)}`);
                        _renderEnvParamSections();
                    } else {
                        updateEnvParamInputs(scenarioParams, 'env_params');
                        if (isClassificationEnv()) {
                            $(`#${CSS.escape('env_params.attacks')}-section`).addClass('hidden');
                            $(`#${CSS.escape('env_params.classification')}-section`).removeClass('hidden');
                        } else {
                            $(`#${CSS.escape('env_params.attacks')}-section`).removeClass('hidden');
                            $(`#${CSS.escape('env_params.classification')}-section`).addClass('hidden');
                        }
                    }
                }
            } catch (err) {
                console.error('Failed to load scenario env params:', err);
            }
            renderDataSourceSelectors();
        });
    }
    // 2. Algorithm Select (Agents only)
    else if (key === 'algorithm' && isAgent) {
        const agentIdx = parseInt(fullId.split('.')[1]);
        inputElement = `<select id="${fullId}" data-path="${fullId}" data-agent-index="${agentIdx}" class="config-select agent-algorithm-select">
                    ${ALGORITHM_OPTIONS.map(opt =>
            `<option value="${opt}" ${value.toLowerCase() === opt.toLowerCase() ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
    }
    // 2b. Load dir modal (Agents only)
    else if (key === 'load_dir' && isAgent) {
        inputElement = `<input type="text" id="${fullId}" data-path="${fullId}" value="${value}" class="config-input load-dir-input" >`;
    }
    // 3. Log level Select
    else if (key === 'log_level') {
        inputElement = `<select id="${fullId}" data-path="${fullId}" class="config-select">
                    ${LOG_LEVELS.map(opt =>
            `<option value="${opt}" ${value.toLowerCase() === opt.toLowerCase() ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
    }
    // 4. Boolean Checkbox
    else if (typeof value === 'boolean') {
        inputElement = `<input type="checkbox" id="${fullId}" data-path="${fullId}" class="w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 config-checkbox ${isHidden ? 'hidden' : ''}" ${value ? 'checked' : ''} ${isReadonly ? 'disabled' : ''}>`;
        //  Enabled (Agents only)
        if (key === 'enabled' && isAgent) {
            $(document).on('change', `#${CSS.escape(fullId)}`, function () {
                const agentIndex = parseInt(fullId.split('.')[1]);
                const agentCard = $(`#agent-card-${agentIndex}`);
                if (Array.isArray(currentConfig.agents) && currentConfig.agents[agentIndex]) {
                    currentConfig.agents[agentIndex].enabled = this.checked;
                }
                if (this.checked) {
                    agentCard.removeClass('opacity-50');
                } else {
                    agentCard.addClass('opacity-50');
                }
                updateAgentTabStyle(agentIndex, this.checked);
                updateAgentsEnabledCount();
            });
        }        
    }
    // 5. State input mode Select
    else if (key === 'state_input_mode') {
        inputElement = `<select id="${fullId}" data-path="${fullId}" class="config-select">
                    ${STATE_INPUT_MODES.map(opt =>
            `<option value="${opt}" ${value.toLowerCase() === opt.toLowerCase() ? 'selected' : ''}>${opt}</option>`
        ).join('')}
                </select>`;
    }    
    // 6. Array (read-only text display)
    else if (Array.isArray(value)) {
        inputElement = `<input type="text" id="${fullId}" data-path="${fullId}" value="${value.join(', ')}" class="config-input" readonly>`;
    }
    // 7. Number or Text
    else {
        const type = typeof value === 'number' ? 'number' : key === 'pwd' ? 'password' : 'text';
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

        inputElement = `<input type="${type}" id="${fullId}" data-path="${fullId}" value="${value}" class="config-input ${isHidden ? 'hidden' : ''} " ${stepAttribute} ${isReadonly ? 'readonly' : ''}>`;

    }

    const mergedNote = [noteElement, commentFromYaml].filter(Boolean).join(' | ');
    // const noteBelowHtml = mergedNote
    //     ? `<p class="text-xs text-gray-500">${escapeHtml(mergedNote)}</p>`
    //     : '';
    const icoInfo = mergedNote ? `<img src="/static/images/icon/info.png" alt="Info" title="${escapeHtml(mergedNote)}" class="inline-block w-4 h-4 ml-1 opacity-50">` : '';
    let fieldHtml = `
                <div class="${isHidden ? 'hidden' : ''} ${typeof value === 'boolean' ? 'flex items-center space-x-3' : 'space-y-1'}">
                    <label for="${fullId}" class="block text-sm font-medium text-gray-700 flex-shrink-0" title="${escapeHtml(mergedNote)} ">${label} ${mergedNote ? icoInfo : ''}</label>
                    ${inputElement}
                </div>
            `;
    return fieldHtml;
}


let list_dir = [];
let selectedSavedConfigPath = '';
let _loadDirSort = { col: 'datetime', asc: false }; // default: newest first

let _datasetList = [];
let _datasetSort = { col: null, asc: true };
let _scenarioList = [];
let _scenarioSort = { col: null, asc: true };

// Parse compact datetime strings used across all tables:
//   "YYYYMMDD-HHMMSS"  (training dirs, load-dir, dataset, scenario)
//   "YYYYMMDD_HHMMSS"  (alternate underscore variant)
//   "YYYY-MM-DD HH:MM:SS"  (saved-configs modified field)
function _parseDatetimeStr(dtStr) {
    const s = String(dtStr || '');
    let m;
    m = s.match(/^(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})$/);
    if (m) return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
    m = s.match(/^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?$/);
    if (m) return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
    const fallback = new Date(s);
    return isNaN(fallback) ? new Date(0) : fallback;
}

// Format any supported datetime string → "DD/MM/YYYY HH:MM"
function _formatDatetimeStr(dtStr) {
    const d = _parseDatetimeStr(dtStr);
    if (!d || isNaN(d.getTime())) return dtStr || '-';
    const pad = n => String(n).padStart(2, '0');
    return `${pad(d.getDate())}/${pad(d.getMonth()+1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// Keep old names as aliases for the load-dir handlers already using them
const _parseDirDatetime  = _parseDatetimeStr;
const _formatDirDatetime = _formatDatetimeStr;

function _sortInd(sortState, col) {
    if (sortState.col !== col) return ' <span class="sort-ind text-gray-400 font-normal text-[10px]">⇅</span>';
    return sortState.asc
        ? ' <span class="sort-ind text-blue-500 font-normal text-[10px]">↑</span>'
        : ' <span class="sort-ind text-blue-500 font-normal text-[10px]">↓</span>';
}

function _getSortVal(item, col) {
    const s = item.summary || {};
    switch (col) {
        case 'datetime': return _parseDatetimeStr(item.datetime).getTime();
        case 'file':           return String(item.file || '').toLowerCase();
        case 'path':           return String(item.path || '').toLowerCase();
        case 'entries':        return Number(s.entries || 0);
        case 'hosts':          return Number(s.hosts || 0);
        case 'status_kinds':   return Number(s.status_kinds || 0);
        case 'attack_like':    return Number(s.attack_like_entries || 0);
        case 'mean_packets':   return Number(s.mean_packets || 0);
        case 'train_episodes': return Number(s.train_episodes || 0);
        case 'train_max_steps':return Number(s.train_max_steps || 0);
        case 'train_steps':    return Number(s.train_steps || 0);
        case 'eval_episodes':  return Number(s.eval_episodes || 0);
        case 'eval_steps':     return Number(s.eval_steps || 0);
        case 'attack_likely':  return String(s.attack_likely_used ?? '').toLowerCase();
        default: return '';
    }
}

function _applySort(list, sortState) {
    if (!sortState.col) return list;
    return [...list].sort((a, b) => {
        const va = _getSortVal(a, sortState.col);
        const vb = _getSortVal(b, sortState.col);
        const cmp = typeof va === 'number' && typeof vb === 'number'
            ? va - vb : String(va).localeCompare(String(vb));
        return sortState.asc ? cmp : -cmp;
    });
}

function renderDatasetRows(list) {
    const selectedPath = currentConfig.env_params.data_traffic_file || '';
    const showAttackCol = !isClassificationEnv();
    const colCount = showAttackCol ? 8 : 7;
    return list.map(item => {
        const summary = item.summary || {};
        const selectedClass = item.path === selectedPath ? 'bg-yellow-500' : '';
        return `
            <tr class="dataset-row border-b hover:bg-gray-50 cursor-pointer ${selectedClass}" data-path="${item.path}">
                <td class="p-2">${_formatDatetimeStr(item.datetime)}</td>
                <!--td class="p-2">${item.file || '-'}</td-->
                <td class="p-2">${summary.entries || 0}</td>
                <td class="p-2">${summary.hosts || 0}</td>
                <td class="p-2">${summary.status_kinds || 0}</td>
                ${showAttackCol ? `<td class="p-2">${summary.attack_like_entries || 0}</td>` : ''}
                <td class="p-2">${summary.mean_packets || 0}</td>
                <td class="p-2 text-xs">${item.path}</td>
            </tr>
        `;
    }).join('') || `<tr><td class="p-2 text-gray-500" colspan="${colCount}">No dataset found</td></tr>`;
}

function renderScenarioRows(list) {
    const selectedScenario = currentConfig.env_params.scenario_file || '';
    const source = currentConfig.env_params.scenario_source || 'generate';
    return list.map(item => {
        const summary = item.summary || {};
        const selectedClass = (source === 'load' && item.path === selectedScenario) ? 'bg-yellow-500' : '';
        const encodedStatistics = encodeURIComponent(JSON.stringify(item.statistics || {}));
        return `
            <tr class="scenario-row border-b hover:bg-gray-50 cursor-pointer ${selectedClass}" data-path="${item.path}" data-statistics="${encodedStatistics}">
                <td class="p-2">${_formatDatetimeStr(item.datetime)}</td>
                <td class="p-2">${summary.train_episodes || 0}</td>
                <td class="p-2">${summary.train_max_steps || 0}</td>
                <td class="p-2">${summary.train_steps || 0}</td>
                <td class="p-2">${summary.eval_episodes || 0}</td>
                <td class="p-2">${summary.eval_steps || 0}</td>
                <td class="p-2">${summary.attack_likely_used ?? '-'}</td>
                <td class="p-2 text-xs">${item.path}</td>
            </tr>
        `;
    }).join('') || '<tr><td class="p-2 text-gray-500" colspan="8">No scenario.json found</td></tr>';
}

function updateLoadSelectedConfigButtonState() {
    const btn = $('#load-selected-config-btn');
    if (!btn.length) return;

    const hasSelection = Boolean(selectedSavedConfigPath);
    btn.prop('disabled', !hasSelection);
    btn.toggleClass('opacity-50 cursor-not-allowed', !hasSelection);
}

function _applyLoadDirSort(list) {
    const { col, asc } = _loadDirSort;
    if (!col) return list;
    return [...list].sort((a, b) => {
        let va, vb;
        if (col === 'datetime') {
            va = _parseDirDatetime(a.datetime).getTime();
            vb = _parseDirDatetime(b.datetime).getTime();
        } else {
            va = a.accuracy;
            vb = b.accuracy;
        }
        return asc ? va - vb : vb - va;
    });
}

function _updateLoadDirSortIndicators() {
    const indDate = _loadDirSort.col === 'datetime'
        ? (_loadDirSort.asc ? ' ↑' : ' ↓') : ' ⇅';
    const indAcc = _loadDirSort.col === 'accuracy'
        ? (_loadDirSort.asc ? ' ↑' : ' ↓') : ' ⇅';
    $('#sort-date').html(`Date<span class="text-xs font-normal text-gray-400">${indDate}</span>`);
    $('#sort-accuracy').html(`Accuracy<span class="text-xs font-normal text-gray-400">${indAcc}</span>`);
}

function renderList(list) {
    const dirListEl = $('#load-dir-list');
    dirListEl.empty();
    if (list.length === 0) {
        dirListEl.append('<p class="p-2 text-gray-500 load-dir-item">No saved training sessions found.</p>');
        return;
    }
    _applyLoadDirSort(list).forEach(dir => {
        const binsLabel = (dir.n_bins != null) ? dir.n_bins : '—';
        const dirItemHtml = `
            <li class="p-2 border-b cursor-pointer hover:bg-gray-100 load-dir-item grid grid-cols-3" title="${escapeHtml(dir.path)}"
                data-n-bins="${dir.n_bins != null ? dir.n_bins : ''}"
                data-episodes="${dir.episodes != null ? dir.episodes : ''}"
                data-algorithm="${escapeHtml(dir.algorithm || '')}">
                <span>${_formatDirDatetime(dir.datetime)}</span>
                <span>${dir.accuracy.toFixed(4)}</span>
                <span>${binsLabel}</span>
                <input type="hidden" value="${dir.path}">
            </li>`;
        dirListEl.append(dirItemHtml);
    });
}

$(document).on('click', '#sort-date', function () {
    if (_loadDirSort.col === 'datetime') {
        _loadDirSort.asc = !_loadDirSort.asc;
    } else {
        _loadDirSort = { col: 'datetime', asc: false }; // default: newest first
    }
    _updateLoadDirSortIndicators();
    renderList(list_dir);
});

$(document).on('click', '#sort-accuracy', function () {
    if (_loadDirSort.col === 'accuracy') {
        _loadDirSort.asc = !_loadDirSort.asc;
    } else {
        _loadDirSort = { col: 'accuracy', asc: false }; // default: highest first
    }
    _updateLoadDirSortIndicators();
    renderList(list_dir);
});


$(document).on('click', '.load-dir-item', function () {
    const targetInputId = $('#load-dir-modal').data('target-input');
    targetCheckId = targetInputId.replace("_dir", '');
    const selectedPath = $(this).find('input[type="hidden"]').val();
    if (!selectedPath) {
        $(`#${CSS.escape(targetInputId)}`).val('');
        $(`#${CSS.escape(targetCheckId)}`).prop('checked', false);
        $('#load-dir-modal').addClass('hidden');
        return;
    }

    $(`#${CSS.escape(targetInputId)}`).val(selectedPath);
    $(`#${CSS.escape(targetCheckId)}`).prop('checked', true);
    $('#load-dir-modal').addClass('hidden');

    const nBins     = $(this).data('n-bins');
    const episodes  = $(this).data('episodes');
    const algorithm = $(this).data('algorithm');

    const rows = [
        ['Path', escapeHtml(selectedPath)],
        ['Algorithm', escapeHtml(algorithm || '—')],
        ['Bins (n_bins)', nBins !== '' ? nBins : '—'],
        ['Episodes', episodes !== '' ? episodes : '—'],
    ];
    const tableRows = rows.map(([k, v]) => `
        <tr class="border-b last:border-0">
            <td class="py-1 pr-4 font-semibold text-gray-600 whitespace-nowrap">${k}</td>
            <td class="py-1 text-gray-800 break-all">${v}</td>
        </tr>`).join('');

    const popupHtml = `
        <div id="agent-info-popup" class="fixed inset-0 bg-gray-900 bg-opacity-60 z-50 flex items-center justify-center">
            <div class="bg-white rounded-xl shadow-2xl p-6 w-full max-w-lg">
                <h3 class="text-lg font-bold mb-4">Agent characteristics</h3>
                <table class="w-full text-sm">${tableRows}</table>
                <button id="close-agent-info-popup" class="mt-5 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 w-full">OK</button>
            </div>
        </div>`;
    $('body').append(popupHtml);
    $('#close-agent-info-popup').on('click', () => $('#agent-info-popup').remove());
    $('#agent-info-popup').on('click', function(e) {
        if (e.target === this) $('#agent-info-popup').remove();
    });
});


$(document).on('click', '.load-dir-input', async function () {
    const fullId = $(this).data('path');
    $('#load-dir-modal').data('target-input', fullId).removeClass('hidden');

    const network_config = $(`#${CSS.escape('env_params.net_params.num_switches')}`).val() + '_' +
        $(`#${CSS.escape('env_params.net_params.num_hosts')}`).val() + '_' +
        $(`#${CSS.escape('env_params.net_params.num_iots')}`).val();

    const gym_type = $(`#${CSS.escape('env_params.gym_type')}`).val();
    const agentIdx = parseInt(fullId.split('.')[1]);
    const agent_name = currentConfig.agents[agentIdx] ? currentConfig.agents[agentIdx].name : '';
    try {
        list_dir = await get_load_dir_list(gym_type, network_config, agent_name);
        _updateLoadDirSortIndicators();
        renderList(list_dir);
    } catch (err) {
        console.error('Failed to load directory list:', err);
    }
});

function get_load_dir_list(gym_type, network_config, agent_name) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/get_load_dir_list?gym_type=' + encodeURIComponent(gym_type) +
                '&network_config=' + encodeURIComponent(network_config) +
                '&agent_name=' + encodeURIComponent(agent_name),
            type: 'GET',
            success: function (response) {
                resolve(response.load_dir_list);
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error load dir list: ' + response.message, 'error');
                reject(response.message);
            }
        });
    });
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
            const sectionComment = resolveConfigComment(currentPath);
            image = '';
            hidden = '';
            if (sectionTitle === "Net Params") {
                image = `<img src="/static/images/gif/network.gif" alt="Network params" title="Network params" class="inline-block w-6 h-6 ">`;
            }
            if (sectionTitle === "Attacks") {
                image = `<img src="/static/images/gif/ddos.gif" alt="Attacks" title="Attacks params" class="inline-block w-6 h-6 ">`;
                if (isClassificationEnv())
                    hidden = 'hidden';
            }
            if (sectionTitle === "Classification") {
                image = `<img src="/static/images/gif/classify.gif" alt="Classification" title="Classification params" class="inline-block w-6 h-6 ">`;
                if (!isClassificationEnv())
                    hidden = 'hidden';
            }            
            html += `
                        <div class="md:col-span-5 bg-gray-50 p-4 rounded-lg border-l-4 border-indigo-500 ${hidden}" id="${fullId}-section">
                            <h5 class="text-md font-semibold text-gray-700 mb-3">${image}${sectionTitle} ${sectionComment ? `<span class="text-xs text-gray-400 mb-3">${escapeHtml(sectionComment)}</span>` : ''}</h5>
                            
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

function _renderEnvParamSections() {
    const UI_SELECTOR_KEYS = new Set(['data_traffic_file', 'scenario_source', 'scenario_file']);
    const SECTION_KEYS = new Set(['attacks', 'classification', 'net_params']);

    // General: flat scalar fields of env_params
    const generalEnv = {};
    for (const [k, v] of Object.entries(currentConfig.env_params)) {
        if (!SECTION_KEYS.has(k) && !UI_SELECTOR_KEYS.has(k)) generalEnv[k] = v;
    }
    $('#env-params-config-list').html(renderFieldsRecursively(generalEnv, 'env_params'));

    // Scenario: attacks / classification sections
    const scenarioEnv = {};
    for (const k of ['attacks', 'classification']) {
        if (k in currentConfig.env_params) scenarioEnv[k] = currentConfig.env_params[k];
    }
    $('#env-scenario-list').html(renderFieldsRecursively(scenarioEnv, 'env_params'));
    if (isClassificationEnv()) {
        $(`#${CSS.escape('env_params.attacks')}-section`).addClass('hidden');
        $(`#${CSS.escape('env_params.classification')}-section`).removeClass('hidden');
    } else {
        $(`#${CSS.escape('env_params.attacks')}-section`).removeClass('hidden');
        $(`#${CSS.escape('env_params.classification')}-section`).addClass('hidden');
    }

    // Network: net_params section
    const networkEnv = currentConfig.env_params.net_params ? { net_params: currentConfig.env_params.net_params } : {};
    $('#env-network-list').html(renderFieldsRecursively(networkEnv, 'env_params'));

    updateEnvScenarioTabLabel();
}

function renderConfig() {
    const render = () => {
    // 1. Render General (Root) Config
    const generalConfig = {};
    ROOT_KEYS.forEach(key => generalConfig[key] = currentConfig[key]);

    // Use 'root' as the path for the top-level items
    const generalHtml = renderFieldsRecursively(generalConfig, 'root');
    $('#general-config-list').html(generalHtml);

    // 2. Render Environment Parameters split into sub-tabs
    _renderEnvParamSections();
    renderDataSourceSelectors();

    // 3. Render Agents
    renderAgents();
    };

    if (!window.configCommentsLoaded) {
        window.configCommentsLoaded = true;
        loadConfigComments().then(render);
        return;
    }

    render();
}

function getCurrentNetworkConfigString() {
    return `${currentConfig.env_params.net_params.num_switches}_${currentConfig.env_params.net_params.num_hosts}_${currentConfig.env_params.net_params.num_iots}`;
}

function shouldShowDatasetSelector() {
    return String(currentConfig.env_params.gym_type || '').includes('_from_dataset');
}

function shouldShowScenarioSelector() {
    const gymType = String(currentConfig.env_params.gym_type || '');
    return gymType.startsWith('attacks_ho') || gymType.startsWith('marl_attacks') || gymType.startsWith('marl_pz');
}

function defaultDatasetPathForGymType(gymType) {
    const baseType = String(gymType || '').replace('_from_dataset', '');
    return `${currentConfig.training_directory}/statuses_${baseType}.json`;
}

function buildSummaryBadge(label, value) {
    return `<span class="inline-block mr-2 mb-2 px-2 py-1 bg-gray-100 rounded text-xs"><b>${label}:</b> ${value}</span>`;
}

function renderScenarioStatsDetails(statistics) {
    if (!statistics || typeof statistics !== 'object' || Object.keys(statistics).length === 0) {
        return '<p class="text-xs text-gray-500">No scenario statistics available.</p>';
    }

    const sections = Object.entries(statistics).map(([sectionName, sectionValue]) => {
        if (!sectionValue || typeof sectionValue !== 'object') {
            return '';
        }

        const rows = Object.entries(sectionValue).map(([k, v]) => {
            const printable = (typeof v === 'object') ? JSON.stringify(v) : String(v);
            return `<div><b>${k}:</b> ${escapeHtml(printable)}</div>`;
        }).join('');

        return `
            <div class="mb-2">
                <div class="text-xs font-semibold text-gray-700 mb-1 capitalize">${escapeHtml(sectionName)}</div>
                <div class="text-xs text-gray-700 grid grid-cols-1 md:grid-cols-3 gap-2">${rows}</div>
            </div>
        `;
    }).join('');

    return `
        <div class="bg-white border rounded-lg p-2 mt-2">
            <div class="text-xs font-semibold text-gray-700 mb-1">Selected Scenario Stats</div>
            ${sections || '<p class="text-xs text-gray-500">No scenario statistics available.</p>'}
        </div>
    `;
}

function renderTestScenarioPreviewPopup(response) {
    const summary = response && response.summary ? response.summary : {};
    const statistics = response && response.statistics ? response.statistics : {};
    const training = statistics.training || {};
    const evaluation = statistics.evaluation || {};
    const envCfg = (currentConfig && currentConfig.env_params) ? currentConfig.env_params : {};
    const attacksCfg = (envCfg && envCfg.attacks) ? envCfg.attacks : {};
    const netCfg = (envCfg && envCfg.net_params) ? envCfg.net_params : {};

    function toNumber(value, fallback = 0) {
        const n = Number(value);
        return Number.isFinite(n) ? n : fallback;
    }

    function pct(value, total) {
        if (!total) return '0.0%';
        return `${((toNumber(value, 0) / total) * 100).toFixed(1)}%`;
    }

    function statusLabel(status) {
        if (status === 'good') return 'Good';
        if (status === 'warn') return 'Watch';
        return 'Low quality';
    }

    function statusColorClass(status) {
        if (status === 'good') return 'bg-green-50 text-green-800 border-green-300';
        if (status === 'warn') return 'bg-yellow-50 text-yellow-800 border-yellow-300';
        return 'bg-red-50 text-red-800 border-red-300';
    }

    function buildQualityTable(assessment) {
        const rows = (assessment && assessment.qualityRows) ? assessment.qualityRows : [];
        if (!rows.length) return '';

        const body = rows.map((row) => {
            const colorClass = statusColorClass(row.status);
            return `
                <tr class="border-b last:border-b-0">
                    <td class="p-1.5 text-left">${escapeHtml(row.label)}</td>
                    <td class="p-1.5 text-right font-medium">${escapeHtml(row.value)}</td>
                    <td class="p-1.5 text-right">
                        <span class="inline-flex items-center gap-1 px-1.5 py-0.5 border rounded text-[10px] font-bold ${colorClass}">
                            <span class="inline-block w-1.5 h-1.5 rounded-full ${row.status === 'good' ? 'bg-green-500' : (row.status === 'warn' ? 'bg-yellow-500' : 'bg-red-500')}"></span>
                            ${escapeHtml(statusLabel(row.status))}
                        </span>
                    </td>
                </tr>
            `;
        }).join('');

        return `
            <div class="overflow-x-auto border-b">
                <table class="w-full text-xs">
                    <thead class="bg-gray-100">
                        <tr>
                            <th class="p-1 text-left">Quality Check</th>
                            <th class="p-1 text-right">Value</th>
                            <th class="p-1 text-right">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${body}
                    </tbody>
                </table>
            </div>
        `;
    }

    function evaluateSection(section, expectedAttackPct) {
        const totalSteps = toNumber(section.total_steps, 0);
        const globalStats = section.global || {};
        const normal = globalStats.normal || {};
        const attack = globalStats.attack || {};

        const attackTotal = toNumber(attack.total, 0);
        const shortAttack = toNumber(attack.short, 0);
        const longAttack = toNumber(attack.long, 0);
        const attackPct = totalSteps > 0 ? (attackTotal * 100.0 / totalSteps) : 0;

        const attackDelta = Math.abs(attackPct - expectedAttackPct);
        let attackStatus = 'bad';
        if (attackDelta <= 8) attackStatus = 'good';
        else if (attackDelta <= 15) attackStatus = 'warn';

        const trafficTypesActive = ['none', 'ping', 'udp', 'tcp']
            .map((key) => toNumber(normal[key], 0))
            .filter((n) => n > 0).length;
        const diversityStatus = trafficTypesActive >= 3 ? 'good' : (trafficTypesActive === 2 ? 'warn' : 'bad');

        const mixStatus = (shortAttack > 0 && longAttack > 0) ? 'good' : ((shortAttack > 0 || longAttack > 0) ? 'warn' : 'bad');

        const statusScore = (status) => (status === 'good' ? 2 : (status === 'warn' ? 1 : 0));
        const score = statusScore(attackStatus) + statusScore(diversityStatus) + statusScore(mixStatus);
        let overall = 'bad';
        if (score >= 5) overall = 'good';
        else if (score >= 3) overall = 'warn';

        const qualityRows = [
            {
                label: 'Attack rate',
                value: `${attackPct.toFixed(1)}% (target ${expectedAttackPct.toFixed(1)}%, Δ ${attackDelta.toFixed(1)}pp)`,
                status: attackStatus,
            },
            {
                label: 'Traffic diversity',
                value: `${trafficTypesActive}/4`,
                status: diversityStatus,
            },
            {
                label: 'Attack mix short/long',
                value: `${shortAttack}/${longAttack}`,
                status: mixStatus,
            },
        ];

        return {
            overall,
            score,
            scoreMax: 6,
            attackPct,
            attackDelta,
            trafficTypesActive,
            qualityRows,
        };
    }

    function sectionRows(section) {
        const totalSteps = toNumber(section.total_steps, 0);
        const globalStats = section.global || {};
        const normal = globalStats.normal || {};
        const attack = globalStats.attack || {};

        const rows = [
            { label: 'Normal total', count: toNumber(normal.total, 0) },
            { label: 'Attack total', count: toNumber(attack.total, 0) },
            { label: 'Short attack', count: toNumber(attack.short, 0) },
            { label: 'Long attack', count: toNumber(attack.long, 0) },
            { label: 'None traffic', count: toNumber(normal.none, 0) },
            { label: 'Ping traffic', count: toNumber(normal.ping, 0) },
            { label: 'UDP traffic', count: toNumber(normal.udp, 0) },
            { label: 'TCP traffic', count: toNumber(normal.tcp, 0) },
        ];

        return rows.map((row) => `
            <tr class="border-b last:border-b-0">
                <td class="p-1 text-left">${escapeHtml(row.label)}</td>
                <td class="p-1 text-right font-medium">${row.count}</td>
                <td class="p-1 text-right text-gray-600">${pct(row.count, totalSteps)}</td>
            </tr>
        `).join('');
    }

    function resolveExpectedAttackPct(section, fallbackPct) {
        const likelyRaw = toNumber(section.attack_likely_used, NaN);
        if (Number.isNaN(likelyRaw)) {
            return fallbackPct;
        }
        const likely = likelyRaw > 1 ? (likelyRaw / 100.0) : likelyRaw;
        return Math.max(0, Math.min(100, likely * 100));
    }

    function buildConfigScenarioTable() {
        const trainGlobal = training.global || {};
        const evalGlobal = evaluation.global || {};
        const trainAttack = trainGlobal.attack || {};
        const evalAttack = evalGlobal.attack || {};
        const trainNormal = trainGlobal.normal || {};
        const evalNormal = evalGlobal.normal || {};

        const hostCount = toNumber(netCfg.num_hosts, 0);
        const iotCount = toNumber(netCfg.num_iots ?? netCfg.num_iot, 0);
        const totalHosts = hostCount + iotCount;

        const configRows = [
            ['Episodes', envCfg.episodes ?? '-'],
            ['Max steps', envCfg.max_steps ?? '-'],
            ['Test episodes', envCfg.test_episodes ?? '-'],
            ['Hosts total (hosts + iots)', `${totalHosts} (${hostCount} + ${iotCount})`],
            ['Max attack %', attacksCfg.max_attack_percentage ?? '-'],
            ['Short duration', attacksCfg.short_attack_duration ?? '-'],
            ['Long duration', attacksCfg.long_attack_duration ?? '-'],
            ['No-attack timeout', attacksCfg.no_attack_timeout ?? '-'],
        ];

        const trainResultRows = [
            ['attack_likely_used', training.attack_likely_used ?? summary.attack_likely_used ?? '-'],
            ['Total steps', summary.train_steps ?? training.total_steps ?? '-'],
            ['Attack total', trainAttack.total ?? '-'],
            ['Normal total', trainNormal.total ?? '-'],
        ];

        const evalResultRows = [
            ['attack_likely_used', evaluation.attack_likely_used ?? '-'],
            ['Total steps', summary.eval_steps ?? evaluation.total_steps ?? '-'],
            ['Attack total', evalAttack.total ?? '-'],
            ['Normal total', evalNormal.total ?? '-'],
        ];

        const makeRows = rows => rows.map(([k, v]) => `
            <tr class="border-b last:border-b-0">
                <td class="p-1 text-left">${escapeHtml(String(k))}</td>
                <td class="p-1 text-right font-medium">${escapeHtml(String(v))}</td>
            </tr>
        `).join('');

        const cfgBody = makeRows(configRows);
        const trainResBody = makeRows(trainResultRows);
        const evalResBody = makeRows(evalResultRows);

        return `
            <div class="border rounded-lg bg-white overflow-hidden">
                <div class="px-2 py-1 bg-gray-50 border-b text-xs font-semibold text-gray-800">Config + Scenario Results</div>
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-0">
                    <div class="overflow-x-auto border-r border-gray-200">
                        <table class="w-full text-xs">
                            <thead class="bg-gray-100">
                                <tr>
                                    <th class="p-1 text-left">Config parameter</th>
                                    <th class="p-1 text-right">Value</th>
                                </tr>
                            </thead>
                            <tbody>${cfgBody}</tbody>
                        </table>
                    </div>
                    <div class="overflow-x-auto border-r border-gray-200">
                        <table class="w-full text-xs">
                            <thead class="bg-blue-50">
                                <tr>
                                    <th class="p-1 text-left text-blue-700">Training scenario</th>
                                    <th class="p-1 text-right text-blue-700">Value</th>
                                </tr>
                            </thead>
                            <tbody>${trainResBody}</tbody>
                        </table>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-xs">
                            <thead class="bg-green-50">
                                <tr>
                                    <th class="p-1 text-left text-green-700">Eval scenario</th>
                                    <th class="p-1 text-right text-green-700">Value</th>
                                </tr>
                            </thead>
                            <tbody>${evalResBody}</tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    function buildSectionCard(title, section, episodesFallback, maxStepsFallback) {
        const totalSteps = toNumber(section.total_steps, 0);
        const episodes = toNumber(section.episodes, episodesFallback);
        const maxSteps = toNumber(section.max_steps, maxStepsFallback);
        const attackLikelyUsed = (section.attack_likely_used ?? summary.attack_likely_used ?? '-');
        const expectedAttackPct = resolveExpectedAttackPct(section, title === 'Training' ? 90.0 : 30.0);
        const assessment = evaluateSection(section, expectedAttackPct);

        const overallLabel = assessment.overall === 'good'
            ? '<span class="text-green-700">Overall quality: Good</span>'
            : (assessment.overall === 'warn'
                ? '<span class="text-yellow-700">Overall quality: Watch</span>'
                : '<span class="text-red-700">Overall quality: Low quality</span>');
        const scoreBadgeClass = assessment.overall === 'good'
            ? 'text-green-800 border-green-300 bg-green-50'
            : (assessment.overall === 'warn'
                ? 'text-yellow-800 border-yellow-300 bg-yellow-50'
                : 'text-red-800 border-red-300 bg-red-50');

        const sectionKey = title === 'Training' ? 'training' : 'evaluation';

        return `
            <div class="border rounded-lg bg-white overflow-hidden h-[43vh] flex flex-col">
                <div class="sticky top-0 z-10 bg-gray-50 border-b">
                    <div class="px-2 py-1">
                    <div class="font-semibold text-gray-800">${escapeHtml(title)}</div>
                    <div class="text-xs text-gray-600">
                        Episodes: <b>${episodes}</b> | Max steps: <b>${maxSteps}</b> | Total steps: <b>${totalSteps}</b> | attack_likely_used: <b>${escapeHtml(String(attackLikelyUsed))}</b>
                    </div>
                    <div class="text-xs font-semibold mt-0.5 flex items-center justify-between gap-2">
                        ${overallLabel}
                        <span class="px-2 py-0.5 rounded border text-xs font-bold ${scoreBadgeClass}">Score ${assessment.score}/${assessment.scoreMax}</span>
                    </div>
                    <div class="text-[11px] text-gray-600">Attack observed: <b>${assessment.attackPct.toFixed(1)}%</b> (target ${expectedAttackPct.toFixed(1)}%)</div>
                    </div>
                </div>
                <div class="flex-1 min-h-0 overflow-y-auto overflow-x-auto">
                    ${buildQualityTable(assessment)}
                    <div class="border-b bg-slate-50 px-2 py-1.5">
                        <div class="text-[11px] font-semibold text-slate-700 mb-1">Attack distribution over time (binned)</div>
                        <div class="h-24">
                            <canvas id="scenario-attack-trend-${sectionKey}"></canvas>
                        </div>
                        <div class="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-slate-700">
                            <span class="inline-flex items-center gap-1 rounded-full bg-red-50 border border-red-200 px-2 py-0.5 font-semibold text-red-700">
                                <span class="inline-block h-2 w-2 rounded-full bg-red-600"></span>
                                Above mean
                            </span>
                            <span class="inline-flex items-center gap-1 rounded-full bg-amber-50 border border-amber-200 px-2 py-0.5 font-semibold text-amber-700">
                                <span class="inline-block h-2 w-2 rounded-full bg-amber-500"></span>
                                Below mean
                            </span>
                            <span class="text-slate-500">Bars are colored relative to the average attack rate for the selected section.</span>
                        </div>
                    </div>
                    <table class="w-full text-xs">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="p-1 text-left">Metric</th>
                                <th class="p-1 text-right">Count</th>
                                <th class="p-1 text-right">% of steps</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${sectionRows(section)}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    const scenarioFilePath = response && response.scenario_file ? String(response.scenario_file) : '';
    const sourceTitle = scenarioFilePath ? 'Scenario file under evaluation' : 'Test scenario generated in memory';
    const sourceDescription = scenarioFilePath
        ? `Scenario path: ${escapeHtml(scenarioFilePath)}`
        : 'No scenario.json file is saved during this preview.';

    return `
        <div class="space-y-1 text-sm">
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-1.5 text-blue-900">
                <div>
                    <div class="font-semibold">${sourceTitle}</div>
                    <div class="text-xs break-all">${sourceDescription}</div>
                </div>
            </div>
            ${buildConfigScenarioTable()}
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-1 items-start">
                <div>
                    ${buildSectionCard('Training', training, summary.train_episodes || 0, summary.train_max_steps || 0)}
                </div>
                <div>
                    ${buildSectionCard('Evaluation', evaluation, summary.eval_episodes || 0, 1)}
                </div>
            </div>
        </div>
    `;
}

function buildAttackTrendBins(series, options = {}) {
    if (!Array.isArray(series) || !series.length) {
        return { labels: [], values: [], tooltipLabels: [] };
    }

    const total = series.length;
    const mode = String(options.mode || 'evaluation');
    const width = Math.max(260, Number(options.width) || 360);
    const episodes = Math.max(0, Number(options.episodes) || 0);
    const maxSteps = Math.max(1, Number(options.maxSteps) || 1);

    const labels = [];
    const tooltipLabels = [];
    const values = [];

    if (mode === 'training' && episodes > 0) {
        const episodesPerBar = width < 520 ? 3 : 2;
        for (let epStart = 0; epStart < episodes; epStart += episodesPerBar) {
            const epEnd = Math.min(epStart + episodesPerBar, episodes);
            const sliceStart = epStart * maxSteps;
            const sliceEnd = Math.min(epEnd * maxSteps, total);
            const slice = series.slice(sliceStart, sliceEnd);
            const attacks = slice.reduce((acc, value) => acc + (Number(value) > 0 ? 1 : 0), 0);
            const pct = slice.length ? (attacks * 100.0 / slice.length) : 0;

            labels.push(`E${epStart + 1}-${epEnd}`);
            tooltipLabels.push(`Episodes ${epStart + 1}-${epEnd} | Steps ${sliceStart + 1}-${sliceEnd}`);
            values.push(Number(pct.toFixed(2)));
        }
        return { labels, values, tooltipLabels };
    }

    const stepChunk = width < 520 ? 10 : 5;
    const safeBins = Math.max(8, Math.min(Math.ceil(total / stepChunk), 80));
    const bucketSize = Math.max(stepChunk, Math.ceil(total / safeBins));

    for (let i = 0; i < total; i += bucketSize) {
        const slice = series.slice(i, i + bucketSize);
        const attacks = slice.reduce((acc, value) => acc + (Number(value) > 0 ? 1 : 0), 0);
        const pct = slice.length ? (attacks * 100.0 / slice.length) : 0;
        const label = `${i + 1}-${Math.min(i + slice.length, total)}`;
        labels.push(label);
        tooltipLabels.push(`Steps ${label}`);
        values.push(Number(pct.toFixed(2)));
    }

    return { labels, values, tooltipLabels };
}

function renderTestScenarioPreviewCharts(response) {
    if (typeof Chart === 'undefined') {
        return;
    }

    const statistics = response && response.statistics ? response.statistics : {};
    const training = statistics.training || {};
    const evaluation = statistics.evaluation || {};

    const sections = [
        {
            key: 'training',
            section: training,
            color: 'rgba(37, 99, 235, 0.85)',
            bg: 'rgba(37, 99, 235, 0.20)',
        },
        {
            key: 'evaluation',
            section: evaluation,
            color: 'rgba(14, 116, 144, 0.90)',
            bg: 'rgba(14, 116, 144, 0.22)',
        },
    ];

    if (!window.__scenarioPreviewCharts) {
        window.__scenarioPreviewCharts = {};
    }

    for (const item of sections) {
        const canvas = document.getElementById(`scenario-attack-trend-${item.key}`);
        if (!canvas) {
            continue;
        }

        const series = Array.isArray(item.section.attack_step_series)
            ? item.section.attack_step_series
            : [];
        const width = (canvas.parentElement && canvas.parentElement.clientWidth)
            ? canvas.parentElement.clientWidth
            : (canvas.clientWidth || 360);
        const episodes = Number(item.section.episodes) || 0;
        const maxSteps = Number(item.section.max_steps) || 1;
        const { labels, values, tooltipLabels } = buildAttackTrendBins(series, {
            mode: item.key,
            width,
            episodes,
            maxSteps,
        });
        const mean = values.length ? values.reduce((acc, v) => acc + v, 0) / values.length : 0;
        const backgroundColors = values.map((v) => (
            v >= mean
                ? 'rgba(185, 28, 28, 0.75)'
                : 'rgba(245, 158, 11, 0.75)'
        ));
        const borderColors = values.map((v) => (
            v >= mean
                ? 'rgba(153, 27, 27, 0.95)'
                : 'rgba(180, 83, 9, 0.95)'
        ));

        const previous = window.__scenarioPreviewCharts[item.key];
        if (previous && typeof previous.destroy === 'function') {
            previous.destroy();
        }

        window.__scenarioPreviewCharts[item.key] = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Attack %',
                    data: values,
                    borderColor: borderColors,
                    backgroundColor: backgroundColors,
                    borderWidth: 1,
                    barPercentage: 1.0,
                    categoryPercentage: 1.0,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (ctx) => tooltipLabels[ctx[0].dataIndex] || `Steps ${ctx[0].label}`,
                            label: (ctx) => `Attack rate: ${ctx.parsed.y.toFixed(1)}% (media ${mean.toFixed(1)}%)`,
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: { display: false },
                        grid: { display: false },
                    },
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            maxTicksLimit: 4,
                            callback: (v) => `${v}%`,
                        },
                        grid: {
                            color: 'rgba(148, 163, 184, 0.2)',
                        },
                    },
                },
            },
        });
    }
}

function renderDatasetTable(datasetList) {
    _datasetList = datasetList;
    const showAttackCol = !isClassificationEnv();
    return `
        <div class="mt-4 bg-gray-50 border rounded-lg p-3">
            <h5 class="text-md font-semibold mb-2">Dataset Source</h5>
            <div class="mb-2">
                ${buildSummaryBadge('Scenario', currentConfig.env_params.gym_type)}
                ${buildSummaryBadge('Default', defaultDatasetPathForGymType(currentConfig.env_params.gym_type))}
            </div>
            <input id="dataset-path-input" data-path="env_params.data_traffic_file" class="config-input mb-2" value="${currentConfig.env_params.data_traffic_file || ''}" />
            <div class="max-h-56 overflow-y-auto border rounded bg-white">
                <table class="w-full text-sm">
                    <thead class="sticky top-0 bg-gray-100">
                        <tr>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="datetime" title="Date and time when the dataset was recorded (YYYYMMDD_HHMMSS)">Date${_sortInd(_datasetSort, 'datetime')}</th>
                            <th class="hidden sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="file" title="Name of the dataset file (statuses_*.json)">File${_sortInd(_datasetSort, 'file')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="entries" title="Total number of traffic status entries (rows) in the dataset">Entries${_sortInd(_datasetSort, 'entries')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="hosts" title="Number of distinct hosts recorded in the dataset">Hosts${_sortInd(_datasetSort, 'hosts')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="status_kinds" title="Number of distinct traffic status types (e.g. normal, syn_flood, …)">Status Kinds${_sortInd(_datasetSort, 'status_kinds')}</th>
                            ${showAttackCol ? `<th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="attack_like" title="Number of entries classified as attack-like traffic">Attack-like${_sortInd(_datasetSort, 'attack_like')}</th>` : ''}
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="mean_packets" title="Average number of packets per entry across all hosts">Mean Packets${_sortInd(_datasetSort, 'mean_packets')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="dataset" data-col="path" title="Full filesystem path to the dataset file">Path${_sortInd(_datasetSort, 'path')}</th>
                        </tr>
                    </thead>
                    <tbody id="dataset-tbody">${renderDatasetRows(_applySort(datasetList, _datasetSort))}</tbody>
                </table>
            </div>
        </div>
    `;
}

function renderScenarioTable(scenarioList) {
    _scenarioList = scenarioList;
    const selectedScenario = currentConfig.env_params.scenario_file || '';
    const source = currentConfig.env_params.scenario_source || 'generate';
    const isGenerateTest = source === 'generate_test';
    const scenarioSourceInfo = resolveConfigComment('ui.scenario_source_help') || [
        'Missing ui.scenario_source_help comment in config_comments.json. Expected to explain the difference between scenario_source options, especially the "generate_test" mode and its use for quick scenario previews without saving scenario.json files.',
    ].join('\n');
    const selectedScenarioItem = scenarioList.find((item) => item.path === selectedScenario) || null;
    const selectedScenarioStatsHtml = selectedScenarioItem
        ? renderScenarioStatsDetails(selectedScenarioItem.statistics || {})
        : '<p class="text-xs text-gray-500 mt-2">Click a row to inspect scenario statistics.</p>';

    return `
        <div class="mt-4 bg-gray-50 border rounded-lg p-3">
            <h5 class="text-md font-semibold mb-2">Scenario Source<img src="/static/images/icon/info.png" alt="Info" title="${escapeHtml(scenarioSourceInfo)}" class="inline-block w-4 h-4 ml-1 opacity-50"></h5>
            <div class="mb-2">
                <label class="mr-3"><input type="radio" name="scenario-source" value="generate" ${source === 'generate' ? 'checked' : ''}> Generate new (default)</label>
                <button type="button" id="scenario-generate-test-btn" class="mr-3 px-2 py-1 text-xs border rounded ${isGenerateTest ? 'bg-blue-100 border-blue-300 text-blue-900' : 'bg-white border-gray-300 text-gray-700'}">Generate test scenario</button>
                <label><input type="radio" name="scenario-source" value="load" ${source === 'load' ? 'checked' : ''}> Load existing</label>
                <input type="hidden" id="scenario-source-hidden" data-path="env_params.scenario_source" value="${source}">
            </div>
            ${isGenerateTest ? '<p class="text-xs text-blue-700 mb-2">Test scenario mode selected: scenario_source=generate_test.</p>' : ''}
            ${source === 'load' ? `
                <div class="grid grid-cols-1 md:grid-cols-5 gap-2 mb-2">
                    <input id="scenario-file-input" data-path="env_params.scenario_file" class="config-input w-full md:col-span-2" value="${selectedScenario}" readonly />
                    <div id="scenario-selected-stats" class="md:col-span-5">${selectedScenarioStatsHtml}</div>
                </div>
            ` : ''}
            <div class="max-h-56 overflow-y-auto border rounded bg-white">
                <table class="w-full text-sm">
                    <thead class="sticky top-0 bg-gray-100">
                        <tr>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="datetime">Date${_sortInd(_scenarioSort, 'datetime')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="train_episodes">Train Eps${_sortInd(_scenarioSort, 'train_episodes')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="train_max_steps">Train Max Steps${_sortInd(_scenarioSort, 'train_max_steps')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="train_steps">Train Steps${_sortInd(_scenarioSort, 'train_steps')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="eval_episodes">Eval Eps${_sortInd(_scenarioSort, 'eval_episodes')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="eval_steps">Eval Steps${_sortInd(_scenarioSort, 'eval_steps')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="attack_likely">Attack likely${_sortInd(_scenarioSort, 'attack_likely')}</th>
                            <th class="sort-th p-2 text-left cursor-pointer hover:bg-gray-200 select-none" data-table="scenario" data-col="path">Path${_sortInd(_scenarioSort, 'path')}</th>
                        </tr>
                    </thead>
                    <tbody id="scenario-tbody">${renderScenarioRows(_applySort(scenarioList, _scenarioSort))}</tbody>
                </table>
            </div>
        </div>
    `;
}

async function renderDataSourceSelectors() {
    const gymType = currentConfig.env_params.gym_type;
    const networkConfig = getCurrentNetworkConfigString();

    // Dataset sub-tab
    const datasetContainer = $('#env-dataset-selectors');
    if (shouldShowDatasetSelector() && datasetContainer.length) {
        if (!currentConfig.env_params.data_traffic_file || currentConfig.env_params.data_traffic_file === 'None') {
            currentConfig.env_params.data_traffic_file = defaultDatasetPathForGymType(gymType);
        }
        const response = await getDatasetList(gymType, networkConfig);
        datasetContainer.html(renderDatasetTable(response));
    } else {
        datasetContainer.html('');
    }

    // Extra sub-tab (scenario generation/loading — attacks_ho / marl only)
    const extraContainer = $('#env-extra-selectors');
    if (shouldShowScenarioSelector() && extraContainer.length) {
        if (!currentConfig.env_params.scenario_source) currentConfig.env_params.scenario_source = 'generate';
        if (!currentConfig.env_params.scenario_file) currentConfig.env_params.scenario_file = '';
        const response = await getScenarioList(gymType, networkConfig);
        extraContainer.html(renderScenarioTable(response));
    } else {
        extraContainer.html('');
    }

    updateEnvSubTabVisibility();
}

function updateEnvScenarioTabLabel() {
    const gymType = (currentConfig && currentConfig.env_params && currentConfig.env_params.gym_type) || '';
    const label = gymType ? `(${gymType})` : '';
    $('#env-scenario-tab-label').text(label);
}

function updateEnvSubTabVisibility() {
    updateEnvScenarioTabLabel();
    const showExtra = shouldShowScenarioSelector();
    const showDataset = shouldShowDatasetSelector();
    $('.env-subtab-extra').toggleClass('hidden', !showExtra);
    $('.env-subtab-dataset').toggleClass('hidden', !showDataset);
    // If the active sub-tab became hidden, fall back to general
    const activeTab = (() => { try { return localStorage.getItem('activeEnvSubTab'); } catch (_) { return null; } })();
    if ((activeTab === 'extra' && !showExtra) || (activeTab === 'dataset' && !showDataset)) {
        switchEnvSubTab('general');
    }
}

$(document).on('click', 'th.sort-th', function () {
    const table = $(this).data('table');
    const col = $(this).data('col');

    if (table === 'dataset') {
        _datasetSort = { col, asc: _datasetSort.col === col ? !_datasetSort.asc : true };
        $('#dataset-tbody').html(renderDatasetRows(_applySort(_datasetList, _datasetSort)));
        $('th.sort-th[data-table="dataset"]').each(function () {
            $(this).find('.sort-ind').replaceWith($(_sortInd(_datasetSort, $(this).data('col')))[0]);
        });
    } else if (table === 'scenario') {
        _scenarioSort = { col, asc: _scenarioSort.col === col ? !_scenarioSort.asc : true };
        $('#scenario-tbody').html(renderScenarioRows(_applySort(_scenarioList, _scenarioSort)));
        $('th.sort-th[data-table="scenario"]').each(function () {
            $(this).find('.sort-ind').replaceWith($(_sortInd(_scenarioSort, $(this).data('col')))[0]);
        });
    }
});

$(document).on('click', '.dataset-row', function () {
    const selectedPath = $(this).data('path');
    $('#dataset-path-input').val(selectedPath);
    currentConfig.env_params.data_traffic_file = selectedPath;
    $('.dataset-row').removeClass('bg-yellow-500');
    $(this).addClass('bg-yellow-500');
    updateConfigurationInMemory();
});

$(document).on('change', '#dataset-path-input', function () {
    currentConfig.env_params.data_traffic_file = $(this).val();
});

$(document).on('change', 'input[name="scenario-source"]', function () {
    const source = $(this).val();
    currentConfig.env_params.scenario_source = source;
    if (source !== 'load') {
        currentConfig.env_params.scenario_file = '';
    }
    renderDataSourceSelectors();
});

$(document).on('click', '#scenario-generate-test-btn', function () {
    currentConfig.env_params.scenario_source = 'generate_test';
    currentConfig.env_params.scenario_file = '';
    updateConfigurationInMemory();
    renderDataSourceSelectors();
    previewTestScenario();
});

function getScenarioDetails(scenarioPath) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/get_scenario_details',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ scenario_path: scenarioPath }),
            success: function (response) {
                resolve(response || {});
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                reject(response.message || 'Unknown error');
            }
        });
    });
}

function inspectLoadedScenario(scenarioPath) {
    getScenarioDetails(scenarioPath)
        .then((response) => {
            const popupHtml = renderTestScenarioPreviewPopup(response || {});
            openInfoPopupHtml(popupHtml, 'Saved Scenario Details', '/static/images/gif/test.gif');
            renderTestScenarioPreviewCharts(response || {});
            showStatus('Scenario details loaded.', 'success');
        })
        .catch((errorMessage) => {
            const message = String(errorMessage || 'Unknown error');
            showStatus('Error loading scenario: ' + message, 'error');
            openInfoPopup('Unable to load scenario details.\n\n' + message);
        });
}

function previewTestScenario() {
    const configToPreview = collectConfigFromUI();
    currentConfig = configToPreview;

    $.ajax({
        url: '/preview_test_scenario',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ config: configToPreview }),
        success: function (response) {
            const popupHtml = renderTestScenarioPreviewPopup(response || {});
            openInfoPopupHtml(popupHtml, 'Test Scenario Preview', '/static/images/gif/test.gif');
            renderTestScenarioPreviewCharts(response || {});
            showStatus('Test scenario preview generated in memory.', 'success');
        },
        error: function (xhr) {
            const response = xhr.responseJSON || { message: xhr.statusText };
            showStatus('Error generating test scenario preview: ' + response.message, 'error');
            openInfoPopup('Unable to generate test scenario preview.\n\n' + (response.message || 'Unknown error'));
        }
    });
}

$(document).on('click', '.scenario-row', function () {
    const selectedPath = $(this).data('path');

    let statistics = {};
    try {
        statistics = JSON.parse(decodeURIComponent($(this).attr('data-statistics') || '{}'));
    } catch (e) {
        statistics = {};
    }

    $('#scenario-file-input').val(selectedPath);
    $('#scenario-file-input').prop('readonly', true);
    $('#scenario-selected-stats').html(renderScenarioStatsDetails(statistics));
    currentConfig.env_params.scenario_source = 'load';
    currentConfig.env_params.scenario_file = selectedPath;
    $('input[name="scenario-source"][value="load"]').prop('checked', true);
    $('#scenario-source-hidden').val('load');
    $('.scenario-row').removeClass('bg-yellow-500');
    $(this).addClass('bg-yellow-500');

    // Sync env params from scenario statistics
    const _training = statistics.training || {};
    const _evaluation = statistics.evaluation || {};
    const _toFinite = (v) => { const n = Number(v); return Number.isFinite(n) ? n : null; };
    const _syncedEpisodes    = _toFinite(_training.episodes);
    const _syncedMaxSteps    = _toFinite(_training.max_steps);
    const _syncedTestEp      = _toFinite(_evaluation.episodes);
    const _syncedLikelyConf  = _toFinite(_training.attack_likely_config);
    const _syncedLikelyTrain = _toFinite(_training.attack_likely_used);
    const _syncedLikelyEval  = _toFinite(_evaluation.attack_likely_used);

    if (_syncedEpisodes !== null) { currentConfig.env_params.episodes = _syncedEpisodes; $('[data-path="env_params.episodes"]').val(_syncedEpisodes); }
    if (_syncedMaxSteps !== null) { currentConfig.env_params.max_steps = _syncedMaxSteps; $('[data-path="env_params.max_steps"]').val(_syncedMaxSteps); }
    if (_syncedTestEp !== null)   { currentConfig.env_params.test_episodes = _syncedTestEp; $('[data-path="env_params.test_episodes"]').val(_syncedTestEp); }
    if (!currentConfig.env_params.attacks) currentConfig.env_params.attacks = {};
    if (_syncedLikelyConf !== null && $('[data-path="env_params.attacks.likely"]').length) {
        currentConfig.env_params.attacks.likely = _syncedLikelyConf;
        $('[data-path="env_params.attacks.likely"]').val(_syncedLikelyConf);
    }
    if (_syncedLikelyTrain !== null && $('[data-path="env_params.attacks.likely_train"]').length) {
        currentConfig.env_params.attacks.likely_train = _syncedLikelyTrain;
        $('[data-path="env_params.attacks.likely_train"]').val(_syncedLikelyTrain);
    }
    if (_syncedLikelyEval !== null && $('[data-path="env_params.attacks.likely_eval"]').length) {
        currentConfig.env_params.attacks.likely_eval = _syncedLikelyEval;
        $('[data-path="env_params.attacks.likely_eval"]').val(_syncedLikelyEval);
    }

    updateConfigurationInMemory();
    renderDataSourceSelectors();
    inspectLoadedScenario(selectedPath);
});

$(document).on('change', '#scenario-file-input', function () {
    currentConfig.env_params.scenario_file = $(this).val();
});

// When user manually edits any of the env params synced from a loaded scenario,
// reset scenario_source back to 'generate' so a new scenario will be generated.
const _SCENARIO_SYNCED_PATHS = new Set([
    'env_params.episodes',
    'env_params.max_steps',
    'env_params.test_episodes',
    'env_params.attacks.likely',
    'env_params.attacks.likely_train',
    'env_params.attacks.likely_eval',
]);

$(document).on('change', '.config-input', function () {
    const path = $(this).data('path');
    if (!_SCENARIO_SYNCED_PATHS.has(path)) return;
    if (!currentConfig.env_params || currentConfig.env_params.scenario_source !== 'load') return;
    currentConfig.env_params.scenario_source = 'generate';
    currentConfig.env_params.scenario_file = '';
    // Sync DOM before collectConfigFromUI() reads it inside updateConfigurationInMemory()
    $('input[name="scenario-source"][value="generate"]').prop('checked', true);
    $('#scenario-source-hidden').val('generate');
    updateConfigurationInMemory();
    renderDataSourceSelectors();
    showStatus('Scenario source reset to "Generate new" after parameter change.', 'info');
});

function reorderEnvParamsForDisplay(envParams) {
    // These keys are managed by custom UI selectors (dataset table, scenario table).
    // Rendering them also as plain inputs causes collectConfigFromUI() to find two inputs
    // with the same data-path, and the stale one (from the initial render) overwrites the
    // value set by the selector, losing the user's selection.
    const UI_SELECTOR_KEYS = new Set(['data_traffic_file', 'scenario_source', 'scenario_file']);

    const SCENARIO_KEYS = ['attacks', 'classification'];
    const present = SCENARIO_KEYS.filter(k => k in envParams);

    if (!present.length) {
        const filtered = {};
        for (const [key, val] of Object.entries(envParams)) {
            if (!UI_SELECTOR_KEYS.has(key)) filtered[key] = val;
        }
        return filtered;
    }

    const ordered = {};
    for (const [key, val] of Object.entries(envParams)) {
        if (SCENARIO_KEYS.includes(key) || UI_SELECTOR_KEYS.has(key)) continue;
        ordered[key] = val;
        if (key === 'gym_type') {
            for (const sk of present) ordered[sk] = envParams[sk];
        }
    }
    return ordered;
}

function getScenarioEnvParams(gymType) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/get_scenario_env_params?gym_type=' + encodeURIComponent(gymType),
            type: 'GET',
            success: function (response) {
                resolve(response);
            },
            error: function (xhr) {
                reject(new Error(xhr.responseText));
            }
        });
    });
}

function updateEnvParamInputs(obj, pathPrefix) {
    for (const key in obj) {
        if (!obj.hasOwnProperty(key)) continue;
        const value = obj[key];
        const fullPath = pathPrefix ? `${pathPrefix}.${key}` : key;
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            updateEnvParamInputs(value, fullPath);
        } else {
            const el = $(`#${CSS.escape(fullPath)}`);
            if (!el.length) continue;
            if (el.attr('type') === 'checkbox') {
                el.prop('checked', Boolean(value));
            } else {
                el.val(value);
            }
        }
    }
}

function getDatasetList(gymType, networkConfig) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/get_dataset_list?gym_type=' + encodeURIComponent(gymType) + '&network_config=' + encodeURIComponent(networkConfig),
            type: 'GET',
            success: function (response) {
                resolve(response.dataset_list || []);
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error loading dataset list: ' + response.message, 'error');
                reject(response.message);
            }
        });
    });
}

function getScenarioList(gymType, networkConfig) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/get_scenario_list?gym_type=' + encodeURIComponent(gymType) + '&network_config=' + encodeURIComponent(networkConfig),
            type: 'GET',
            success: function (response) {
                resolve(response.scenario_list || []);
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error loading scenario list: ' + response.message, 'error');
                reject(response.message);
            }
        });
    });
}

function renderSavedConfigsTable(configList) {
    const tbody = $('#saved-configs-table-body');
    tbody.empty();

    let hasSelectedRow = false;
    if (!configList || configList.length === 0) {
        selectedSavedConfigPath = '';
        updateLoadSelectedConfigButtonState();
        tbody.append('<tr><td class="p-2 text-gray-500" colspan="6">No saved configs found</td></tr>');
        return;
    }

    configList.forEach((cfg) => {
        const isSelected = cfg.path === selectedSavedConfigPath;
        if (isSelected) hasSelectedRow = true;
        const selectedClass = isSelected ? 'bg-blue-100' : '';
        tbody.append(`
            <tr class="saved-config-row border-b hover:bg-gray-50 cursor-pointer ${selectedClass}" data-path="${cfg.path}">
                <td class="p-2">${_formatDatetimeStr(cfg.modified)}</td>
                <td class="p-2">${cfg.gym_type || '-'}</td>
                <td class="p-2">${cfg.network_config || '-'}</td>
                <td class="p-2">${cfg.episodes || 0}/${cfg.test_episodes || 0}</td>
                <td class="p-2">${cfg.enabled_agents || 0}</td>
                <td class="p-2 text-xs">${cfg.path}</td>
            </tr>
        `);
    });

    if (!hasSelectedRow) {
        selectedSavedConfigPath = '';
    }
    updateLoadSelectedConfigButtonState();
}

function loadSavedConfigsTable() {
    updateLoadSelectedConfigButtonState();
    $.ajax({
        url: '/get_saved_configs_list',
        type: 'GET',
        success: function (response) {
            renderSavedConfigsTable(response.config_list || []);
        },
        error: function (xhr) {
            const response = xhr.responseJSON || { message: xhr.statusText };
            showStatus('Error loading saved configs: ' + response.message, 'error');
        }
    });
}

$(document).on('click', '.saved-config-row', function () {
    selectedSavedConfigPath = $(this).data('path');
    $('.saved-config-row').removeClass('bg-blue-100');
    $(this).addClass('bg-blue-100');
    updateLoadSelectedConfigButtonState();
});

function loadSelectedSavedConfig() {
    if (!selectedSavedConfigPath) {
        alert('Please select a saved configuration before loading.');
        showStatus('Select a saved configuration first.', 'info');
        return;
    }

    $.ajax({
        url: '/load_saved_config',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ path: selectedSavedConfigPath }),
        success: function (response) {
            currentConfig = response.config;
            renderConfig();
            updateConfigurationInMemory();
            showStatus('Saved configuration loaded successfully.', 'success');
            $('#config-list-modal').addClass('hidden');
            selectedSavedConfigPath = '';
            updateLoadSelectedConfigButtonState();
        },
        error: function (xhr) {
            const response = xhr.responseJSON || { message: xhr.statusText };
            showStatus('Error loading saved config: ' + response.message, 'error');
        }
    });
}

function renderAgents() {
    const agentsListEl = $('#agents-list');
    const tabBarEl = $('#agents-tab-bar');
    agentsListEl.empty();
    tabBarEl.empty();

    if (!currentConfig.agents.length) {
        agentsListEl.html('<p class="text-sm text-gray-400 italic p-2">No agents configured. Add one above.</p>');
        updateAgentsEnabledCount();
        return;
    }

    let tabBarHtml = '';
    let panelsHtml = '';

    currentConfig.agents.forEach((agent, index) => {
        const enabled = agent.enabled !== false;
        const dotClass = enabled ? 'agent-dot-on' : 'agent-dot-off';
        const tabDisabled = enabled ? '' : 'agent-tab-disabled';

        tabBarHtml += `<button class="agent-tab ${tabDisabled}" data-agent-tab="${index}">
            <span class="agent-dot ${dotClass}"></span>
            <span id="agent-tab-label-${index}">${escapeHtml(agent.name || `Agent ${index}`)}</span>
        </button>`;

        panelsHtml += `<div class="agent-tab-panel hidden" id="agent-panel-${index}">
            <div id="agent-card-${index}" class="bg-gray-50 p-4 rounded-xl border border-gray-200 shadow-md ${enabled ? '' : 'opacity-50'}">
                <div class="flex justify-between items-start mb-3 border-b pb-2">
                    <h4 id="agents.${index}.title" class="text-lg font-semibold text-gray-800">${escapeHtml(agent.name || '')}</h4>
                    <div class="flex items-center gap-2">
                        <button data-agent-name="${escapeAttr(agent.name)}" class="duplicate-agent-btn text-blue-400 hover:text-blue-600 transition duration-150" title="Duplicate agent">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                        </button>
                        <button data-agent-name="${escapeAttr(agent.name)}" class="remove-agent-btn text-red-500 hover:text-red-700 transition duration-150" title="Remove agent">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                        </button>
                    </div>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-6 gap-4">
                    ${renderFieldsRecursively(agent, `agents.${index}`, true, index)}
                </div>
            </div>
        </div>`;
    });

    tabBarEl.html(tabBarHtml);
    agentsListEl.html(panelsHtml);

    const saved = (() => { try { return parseInt(localStorage.getItem('activeAgentTab') || '0'); } catch (_) { return 0; } })();
    const validIdx = (saved >= 0 && saved < currentConfig.agents.length) ? saved : 0;
    switchAgentTab(validIdx);

    updateAgentsEnabledCount();
}

function switchAgentTab(index) {
    $('.agent-tab-panel').addClass('hidden');
    $(`#agent-panel-${index}`).removeClass('hidden');
    $('.agent-tab').removeClass('agent-tab-active');
    $(`.agent-tab[data-agent-tab="${index}"]`).addClass('agent-tab-active');
    try { localStorage.setItem('activeAgentTab', String(index)); } catch (_) {}
}

function updateAgentTabStyle(index, enabled) {
    const tab = $(`.agent-tab[data-agent-tab="${index}"]`);
    tab.toggleClass('agent-tab-disabled', !enabled);
    tab.find('.agent-dot').toggleClass('agent-dot-on', enabled).toggleClass('agent-dot-off', !enabled);
}

$(document).on('click', '.agent-tab', function () {
    switchAgentTab(parseInt($(this).data('agent-tab')));
});

function updateAgentsEnabledCount() {
    const counterEl = $('#agents_enabled');
    if (!counterEl.length) {
        return;
    }

    const enabledCount = Array.isArray(currentConfig.agents)
        ? currentConfig.agents.filter((agent) => agent && agent.enabled).length
        : 0;
    counterEl.text(enabledCount);
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
    const selectedAlgo = $('#add-agent-algo-select').val() || 'DQN';
    const defaults = algoDefaults[selectedAlgo.toLowerCase()] || { algorithm: selectedAlgo, enabled: true };
    const newAgentName = "New_Agent_" + (currentConfig.agents.length + 1);
    const newAgent = Object.assign({}, defaults, { name: newAgentName, enabled: true });
    currentConfig.agents.push(newAgent);
    renderAgents();
    showStatus(`Agent "${newAgentName}" added with ${selectedAlgo} defaults.`, 'info');
}

function duplicateAgent(agentName) {
    const source = currentConfig.agents.find(a => a.name === agentName);
    if (!source) return;
    const copy = JSON.parse(JSON.stringify(source));
    copy.name = agentName + '_copy';
    currentConfig.agents.push(copy);
    renderAgents();
    showStatus(`Agent "${agentName}" duplicated as "${copy.name}".`, 'success');
}

function applyAlgorithmDefaults(agentIndex, newAlgo) {
    const defaults = algoDefaults[newAlgo.toLowerCase()];
    if (!defaults) {
        showStatus(`No defaults found for algorithm "${newAlgo}".`, 'error');
        return;
    }
    const current = currentConfig.agents[agentIndex];
    const name = current.name;
    const enabled = current.enabled;
    currentConfig.agents[agentIndex] = Object.assign({}, defaults, { name, enabled });
    renderAgents();
    showStatus(`Agent "${name}": switched to ${newAlgo} — defaults applied.`, 'info');
}

function removeAgent(agentName) {
    const initialLength = currentConfig.agents.length;
    currentConfig.agents = currentConfig.agents.filter(agent => agent.name !== agentName);
    if (currentConfig.agents.length < initialLength) {
        renderAgents();
        showStatus(`Agent ${agentName} removed.`, 'success');
    }
}

function selectAllAgents(enabled) {
    currentConfig.agents.forEach(agent => {
        agent.enabled = enabled;
    });
    renderAgents();
    showStatus(enabled ? 'All agents enabled.' : 'All agents disabled.', 'info');
}

// ====================================================================\
// FILE HANDLING
// ====================================================================\


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
            renderConfig();
            showStatus('YAML configuration loaded successfully.', 'success');
            updateConfigurationInMemory();
            $('#config-list-modal').addClass('hidden');
        } catch (error) {
            alert('Error parsing YAML file: ' + error.message);
            showStatus('Error parsing YAML file: ' + error.message, 'error');
        }
    };

    reader.onerror = function () {
        const errorMessage = reader.error ? reader.error.message : 'Unknown file read error';
        showStatus('Error reading YAML file: ' + errorMessage, 'error');
    };

    reader.readAsText(file);
}

function closeResultModal() {
    $('#result-modal').addClass('hidden');
}

function renderConfigSummary() {
    const env = currentConfig.env_params || {};
    const classification = isClassificationEnv();

    // --- Environment ---
    const envEl = $('#config-env');
    envEl.empty();
    envEl.append('<h4 class="text-lg font-semibold text-gray-800 mb-3"><img src="static/images/gif/earth.gif" alt="Environment Parameters" title="Environment Parameters" class="inline-block w-6 h-6 mr-1">Environment Parameters</h4>');

    const generalParams = {};
    ['gym_type', 'episodes', 'max_steps', 'test_episodes', 'n_bins',
     'steps_min_percentage', 'accuracy_min', 'show_normal_traffic',
     'print_training_chart', 'must_check_env', 'wait_after_read'].forEach(k => {
        if (k in env) generalParams[k] = env[k];
    });
    envEl.append(_renderKVSection('General', generalParams));

    if (classification && env.classification) {
        envEl.append(_renderKVSection('Classification Settings', env.classification));
    } else if (!classification && env.attacks) {
        envEl.append(_renderKVSection('Attack Settings', env.attacks));
    }

    if (env.net_params) {
        envEl.append(_renderKVSection('Network', env.net_params));
    }

    // --- Agents (only enabled) ---
    const agentsEl = $('#config-agents');
    agentsEl.empty();
    agentsEl.append('<h4 class="text-lg font-semibold text-gray-800 mb-3"><img src="static/images/gif/manager.gif" alt="Agents" title="Agents" class="inline-block w-6 h-6 mr-1">Active Agents</h4>');

    const activeAgents = (Array.isArray(currentConfig.agents) ? currentConfig.agents : [])
        .filter(a => a.enabled === true || a.enabled === 'true');

    if (activeAgents.length === 0) {
        agentsEl.append('<p class="text-sm text-gray-500 italic">No agents enabled.</p>');
    } else {
        activeAgents.forEach(agent => {
            const params = {};
            Object.keys(agent).forEach(k => {
                if (k !== 'name' && k !== 'enabled' && k !== 'algorithm') params[k] = agent[k];
            });
            const title = `${agent.name} <span class="font-normal text-gray-500">(${agent.algorithm || ''})</span>`;
            agentsEl.append(_renderKVSection(title, params));
        });
    }
}

function _renderKVSection(title, data) {
    const section = $('<div class="mb-3 bg-gray-50 rounded-lg p-3 border border-gray-200"></div>');
    section.append(`<h5 class="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2 border-b border-gray-200 pb-1">${title}</h5>`);

    const activeGrid = $('<div class="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs"></div>');
    const inactiveGrid = $('<div class="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs hidden mt-1 pt-1 border-t border-dashed border-gray-300 opacity-60"></div>');

    _renderKVPairs(data, activeGrid, inactiveGrid);
    section.append(activeGrid);

    const inactiveCount = inactiveGrid.find('[data-kv-val]').length;
    if (inactiveCount > 0) {
        const btn = $(`<button class="mt-1.5 text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1 select-none"><span class="arr">▶</span><span class="lbl">${inactiveCount} disabled</span></button>`);
        btn.on('click', function () {
            const opening = inactiveGrid.hasClass('hidden');
            inactiveGrid.toggleClass('hidden');
            $(this).find('.arr').text(opening ? '▼' : '▶');
            $(this).find('.lbl').text(opening ? 'hide' : `${inactiveCount} disabled`);
        });
        section.append(btn);
        section.append(inactiveGrid);
    }

    return section;
}

function _renderKVPairs(data, activeEl, inactiveEl) {
    Object.keys(data).forEach(key => {
        const val = data[key];
        if (val === undefined) return;

        if (val !== null && typeof val === 'object' && !Array.isArray(val)) {
            const subActive = $('<div></div>');
            const subInactive = $('<div></div>');
            _renderKVPairs(val, subActive, subInactive);
            if (subActive.children().length > 0) {
                activeEl.append(`<div class="col-span-2 mt-2 mb-0.5 font-semibold text-indigo-600">${key}</div>`);
                activeEl.append(subActive.children());
            }
            if (subInactive.children().length > 0) {
                inactiveEl.append(`<div class="col-span-2 mt-2 mb-0.5 font-semibold text-indigo-600">${key}</div>`);
                inactiveEl.append(subInactive.children());
            }
        } else {
            const isInactive = val === false || val === null || val === 'false' || val === 'null' || val === '';
            const displayVal = Array.isArray(val) ? JSON.stringify(val) : String(val ?? '—');
            const target = isInactive ? inactiveEl : activeEl;
            target.append(`<div class="text-gray-500 py-0.5 truncate" title="${key}">${key}</div>`);
            target.append(`<div class="text-gray-800 font-medium py-0.5 truncate" data-kv-val title="${displayVal}">${displayVal}</div>`);
        }
    });
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


// ── Config tab switching ──────────────────────────────────────────────────────

function switchConfigTab(tabName) {
    $('.config-tab-panel').addClass('hidden');
    $(`#tab-${tabName}`).removeClass('hidden');
    $('.config-tab').removeClass('cfg-tab-active');
    $(`.config-tab[data-tab="${tabName}"]`).addClass('cfg-tab-active');
    try { localStorage.setItem('activeConfigTab', tabName); } catch (_) {}
}

function initConfigTabs() {
    const saved = (() => { try { return localStorage.getItem('activeConfigTab'); } catch (_) { return null; } })();
    switchConfigTab(saved || 'env');
}

$(document).on('click', '.config-tab', function () {
    switchConfigTab($(this).data('tab'));
});

// ── Env sub-tab switching ─────────────────────────────────────────────────────

function switchEnvSubTab(tabName) {
    $('.env-sub-panel').addClass('hidden');
    $(`#env-sub-${tabName}`).removeClass('hidden');
    $('.env-sub-tab').removeClass('env-subtab-active');
    $(`.env-sub-tab[data-env-tab="${tabName}"]`).addClass('env-subtab-active');
    try { localStorage.setItem('activeEnvSubTab', tabName); } catch (_) {}
}

function initEnvSubTabs() {
    const saved = (() => { try { return localStorage.getItem('activeEnvSubTab'); } catch (_) { return null; } })();
    switchEnvSubTab(saved || 'general');
}

$(document).on('click', '.env-sub-tab', function () {
    switchEnvSubTab($(this).data('env-tab'));
});

// ── Theme toggle ──────────────────────────────────────────────────────────────

function toggleTheme() {
    const isDark = document.documentElement.hasAttribute('data-theme');
    if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        try { localStorage.setItem('theme', 'light'); } catch (_) {}
        $('#theme-icon').html(_sunIcon());
    } else {
        document.documentElement.setAttribute('data-theme', 'cyber');
        try { localStorage.setItem('theme', 'cyber'); } catch (_) {}
        $('#theme-icon').html(_moonIcon());
    }
}

function initTheme() {
    const saved = (() => { try { return localStorage.getItem('theme'); } catch (_) { return null; } })();
    if (saved === 'light') {
        document.documentElement.removeAttribute('data-theme');
        $('#theme-icon').html(_sunIcon());
    } else {
        document.documentElement.setAttribute('data-theme', 'cyber');
        $('#theme-icon').html(_moonIcon());
    }
}

function _moonIcon() {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M21 12.79A9 9 0 1111.21 3a7 7 0 109.79 9.79z"/></svg>`;
}

function _sunIcon() {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
}
