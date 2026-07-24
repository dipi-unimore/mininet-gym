

// ====================================================================\
// SESSION STATE MANAGEMENT (Training Data Persistence)
// ====================================================================\

// Helper to save training data to sessionStorage for recovery on page reload
function saveTrainingSessionData() {
    try {
        const sessionData = {
            lastAgentTrainingSummaries: window.lastAgentTrainingSummaries || {},
            lastAgentEvaluationSummaries: window.lastAgentEvaluationSummaries || {},
            lastAgentStepStats: window.lastAgentStepStats || {},
            lastHostTasks: window.lastHostTasks || {},
            lastAgentStatuses: window.lastAgentStatuses || {},
            timestamp: Date.now()
        };
        sessionStorage.setItem('trainingSessionData', JSON.stringify(sessionData));
    } catch (e) {
        console.warn('Could not save training session data:', e);
    }
}

// Helper to restore training data from sessionStorage
function restoreTrainingSessionData() {
    try {
        const stored = sessionStorage.getItem('trainingSessionData');
        if (!stored) return false;
        
        const sessionData = JSON.parse(stored);
        
        // Restore data to global variables
        window.lastAgentTrainingSummaries = sessionData.lastAgentTrainingSummaries || {};
        window.lastAgentEvaluationSummaries = sessionData.lastAgentEvaluationSummaries || {};
        window.lastAgentStepStats = sessionData.lastAgentStepStats || {};
        window.lastHostTasks = sessionData.lastHostTasks || {};
        window.lastAgentStatuses = sessionData.lastAgentStatuses || {};
        
        return true;
    } catch (e) {
        console.warn('Could not restore training session data:', e);
        return false;
    }
}

// Restore UI states for training results buttons
function restoreTrainingResultsButtonStates() {
    if (!window.lastAgentTrainingSummaries) return;
    
    for (const agentName in window.lastAgentTrainingSummaries) {
        const button = $(`.agent-training-popup-btn[data-agent="${agentName}"]`).first();
        if (button.length) {
            button.prop('disabled', false)
                .attr('data-ready', 'true')
                .removeClass('bg-slate-200 text-slate-500 cursor-not-allowed')
                .addClass('bg-blue-600 text-white hover:bg-blue-700 cursor-pointer')
                .attr('title', 'Open training result summary');
        }
    }
}

// Restore UI states for evaluation results buttons
function restoreEvaluationResultsButtonStates() {
    if (!window.lastAgentEvaluationSummaries) return;
    
    for (const agentName in window.lastAgentEvaluationSummaries) {
        const button = $(`.agent-evaluation-popup-btn[data-agent="${agentName}"]`).first();
        if (button.length) {
            button.prop('disabled', false)
                .attr('data-ready', 'true')
                .removeClass('bg-slate-200 text-slate-500 cursor-not-allowed')
                .addClass('bg-amber-500 text-white hover:bg-amber-600 cursor-pointer')
                .attr('title', 'Open evaluation result summary');
        }
    }
}

// ====================================================================\
// EVENT LISTENERS & INITIALIZATION
// ====================================================================\

// Helper function to save training state to localStorage
function saveTrainingStateToStorage(status) {
    try {
        localStorage.setItem('trainingStatus', JSON.stringify({
            status: status,
            timestamp: Date.now()
        }));
    } catch (e) {
        console.warn('Could not save training status to localStorage:', e);
    }
}

// Helper function to get training state from localStorage
function getTrainingStateFromStorage() {
    try {
        const stored = localStorage.getItem('trainingStatus');
        return stored ? JSON.parse(stored) : null;
    } catch (e) {
        console.warn('Could not retrieve training status from localStorage:', e);
        return null;
    }
}

// Check if there's an active training and navigate automatically
async function checkAndRecoverTrainingState() {
    try {
        const response = await $.ajax({
            url: '/get_training_status',
            type: 'GET',
            timeout: 5000
        });
        
        if (response.is_training) {
            console.log('Active training detected:', response.message);
            saveTrainingStateToStorage(response.status);

            // Sync systemStatus immediately from the AJAX response (mirrors
            // syncTrainingStateOnReconnect's mapping) instead of waiting for
            // the socket to (re)connect — this also locks the Configuration
            // page right away via setStatus() -> updateConfigLockFromStatus().
            let recoveredState = SYSTEM_STATUS.TRAINING_RUNNING;
            if (response.is_paused) {
                recoveredState = SYSTEM_STATUS.PAUSED;
            } else if (response.is_stopping) {
                recoveredState = SYSTEM_STATUS.STOPPED;
            }
            setStatus(recoveredState, response.message);

            // Stash which agent the server says is currently training/evaluating
            // so the tab bar (built later, during navigation) can mark it
            // 'Running' immediately instead of waiting for the next live metric.
            window.pendingCurrentTrainingAgent = response.current_agent || null;
            // Stash cached comm/sync events so the tab bar (built later) can
            // replay them into the per-host timelines (see agent_statistics.js).
            window.pendingAgentCommEvents = response.agent_comm_events || null;

            // Restore training session data (summaries, stats, etc.)
            const dataRestored = restoreTrainingSessionData();
            
            // Restore chart data
            let chartDataRestored = false;
            if (typeof restoreChartDataFromSession === 'function' && response.agent_chart_data) {
                chartDataRestored = restoreChartDataFromSession(response.agent_chart_data);
            }

            // Populate summaries from server (actual data, not placeholders) so
            // restoreTrainingResultsButtonStates / restoreEvaluationResultsButtonStates
            // can enable the appropriate buttons with working popup data.
            if (response.agent_button_state) {
                window.lastAgentTrainingSummaries = window.lastAgentTrainingSummaries || {};
                window.lastAgentEvaluationSummaries = window.lastAgentEvaluationSummaries || {};
                Object.keys(response.agent_button_state).forEach(agent => {
                    const st = response.agent_button_state[agent] || {};
                    if (st.training_summary && !window.lastAgentTrainingSummaries[agent]) {
                        window.lastAgentTrainingSummaries[agent] = st.training_summary;
                    }
                    if (st.evaluation_summary && !window.lastAgentEvaluationSummaries[agent]) {
                        window.lastAgentEvaluationSummaries[agent] = st.evaluation_summary;
                    }
                });
            }

            if (dataRestored || chartDataRestored) {
                console.log('Restored previous training data');
            }
            
            // Navigate to training page automatically
            await navigateTo('training');
            return true;
        } else {
            // Clear any stale training state
            window.pendingCurrentTrainingAgent = null;
            window.pendingAgentCommEvents = null;
            localStorage.removeItem('trainingStatus');
            sessionStorage.removeItem('trainingSessionData');
            sessionStorage.removeItem('chartDataRaw');
            return false;
        }
    } catch (error) {
        console.warn('Could not check training status:', error);
        // If we can't reach the server, check localStorage as fallback
        const storedState = getTrainingStateFromStorage();
        if (storedState && storedState.status) {
            // Restore session data as well
            restoreTrainingSessionData();
            if (typeof restoreChartDataFromSession === 'function') {
                restoreChartDataFromSession();
            }
            // Assume training was active, navigate there anyway
            await navigateTo('training');
            return true;
        }
        return false;
    }
}

$(document).ready(async function () {
    // Load algorithm defaults from backend
    $.ajax({
        url: '/get_algo_defaults',
        type: 'GET',
        success: function (data) {
            algoDefaults = data;
        },
        error: function () {
            console.warn('Could not load algorithm defaults');
        }
    });

    // Check for active training BEFORE initial render
    const wasRecovered = await checkAndRecoverTrainingState();

    // Initial render of the configuration
    renderConfig();
    initConfigTabs();
    initEnvSubTabs();
    initTheme();
    
    // Navigate to appropriate page (training if recovered, otherwise config)
    if (!wasRecovered) {
        navigateTo('config'); // Start on the config page if no active training
    } else {
        // If we recovered training, restore UI states for result buttons
        setTimeout(() => {
            restoreTrainingResultsButtonStates();
            restoreEvaluationResultsButtonStates();
        }, 500); // Small delay to ensure DOM is ready
    }

    $(document).ajaxStart(function () {
        showLoadingOverlay('ajax', 'Loading data...');
    });

    $(document).ajaxStop(function () {
        hideLoadingOverlay('ajax');
    });

    // --- Configuration Page Buttons ---
    $('#save-config-btn').click(saveConfiguration);

    $('#download-result-btn').click(downloadResults);

    // Sync agent card title and tab label when name field is edited
    $('#agents-list').on('input', 'input[id^="agents."][id$=".name"]', function () {
        const agentIndex = $(this).attr('id').split('.')[1];
        const name = $(this).val();
        $(`#${CSS.escape('agents.' + agentIndex + '.title')}`).text(name);
        $(`#agent-tab-label-${agentIndex}`).text(name);
    });

    // Listener for the remove agent buttons (uses event delegation)
    $('#agents-list').on('click', '.remove-agent-btn', function (e) {
        e.preventDefault();
        const agentName = $(this).data('agent-name');
        removeAgent(agentName);
    });

    // Listener for duplicate agent buttons
    $('#agents-list').on('click', '.duplicate-agent-btn', function (e) {
        e.preventDefault();
        const agentName = $(this).data('agent-name');
        duplicateAgent(agentName);
    });

    // Listener for algorithm dropdown change — replace agent params with defaults
    $('#agents-list').on('change', '.agent-algorithm-select', function () {
        const agentIndex = parseInt($(this).data('agent-index'));
        const newAlgo = $(this).val();
        applyAlgorithmDefaults(agentIndex, newAlgo);
    });

    $('#add-agent-btn').click(addAgent);
    $('#select-all-agents-btn').click(function() {
        selectAllAgents(true);
    });
    $('#deselect-all-agents-btn').click(function() {
        selectAllAgents(false);
    });

    // --- Training Page Buttons ---

    // Start Training
    $('#start-training-btn').click(function () {
        if (systemStatus==SYSTEM_STATUS.TRAINING_STARTING ||
            systemStatus==SYSTEM_STATUS.TRAINING_RUNNING ||
            systemStatus==SYSTEM_STATUS.PLOTTING_TRAINING_DATA ||
            systemStatus==SYSTEM_STATUS.EVALUATING_RUNNING
        ) return;
        showStatus('Attempting to start training...', 'info');

        $.ajax({
            url: '/start_training',
            type: 'POST',
            contentType: 'application/json',
            success: function (response) {
                showStatus(response.message, 'success');
                if (response.status == STATUS.STARTING) {
                    if (typeof resetTrainingDashboardForNewRun === 'function') {
                        resetTrainingDashboardForNewRun();
                    }
                    setStatus(SYSTEM_STATUS.TRAINING_STARTING, response.message );
                    saveTrainingStateToStorage(SYSTEM_STATUS.TRAINING_STARTING);
                    showCharts();
                } else {
                    setStatus(SYSTEM_STATUS.RESUMED, response.message );
                    saveTrainingStateToStorage(SYSTEM_STATUS.RESUMED);
                }
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error starting training: ' + response.message, 'error');
                setStatus(SYSTEM_STATUS.ERROR, 'Error starting training');
            }
        });
    });

    // Pause Training
    $('#pause-training-btn').click(function () {
        if ((systemStatus!=SYSTEM_STATUS.TRAINING_RUNNING 
            && systemStatus!=SYSTEM_STATUS.TRAINING_STARTING
            && systemStatus!=SYSTEM_STATUS.RESUMED
            && systemStatus!=SYSTEM_STATUS.EVALUATING_RUNNING)
            || systemStatus==SYSTEM_STATUS.PAUSED
            || systemStatus==SYSTEM_STATUS.STOPPED
        ) return;

        $.ajax({
            url: '/pause_training',
            type: 'POST',
            success: function (response) {
                setStatus(SYSTEM_STATUS.PAUSED, 'Training paused.');
                saveTrainingStateToStorage(SYSTEM_STATUS.PAUSED);
                saveTrainingSessionData();
                showStatus(response.message || 'Training paused.', 'info');
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error pausing training: ' + response.message, 'error');
                setStatus(SYSTEM_STATUS.ERRO, 'Error pausing training');
            }
        });
    });

    // Stop Training
    $('#stop-training-btn').click(function () {
        if (systemStatus==SYSTEM_STATUS.STOPPED) return;
        showStatus('Attempting to stop training...', 'info');

        $.ajax({
            url: '/stop_training',
            type: 'POST',
            success: function (response) {
                if (socket && socket.connected) {
                    socket.close();
                }
                setStatus(SYSTEM_STATUS.STOPPED, 'Training session stopped.');
                localStorage.removeItem('trainingStatus');
                sessionStorage.removeItem('trainingSessionData');
                showStatus(response.message || 'Training session stopped.', 'success');
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error stopping training: ' + response.message, 'error');
                setStatus(SYSTEM_STATUS.ERROR, 'Error stopping training');
            }
        });
    });

    // --- Threshold Calibration (System tab) ---
    let calibrationPollTimer = null;

    function renderCalibrationLog(lines) {
        const $log = $('#calibration-log');
        $log.removeClass('hidden').text((lines || []).join('\n'));
        $log.scrollTop($log[0].scrollHeight);
    }

    function fmtNum(n) {
        return Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
    }

    function renderCalibrationResult(result) {
        if (!result) return;
        const rows = ['packets', 'bytes'].map(metric => {
            const s = result[metric];
            const ratioColor = s.ok ? 'text-green-600' : 'text-red-600';
            return `
                <div class="mt-2">
                    <div class="font-semibold">${metric === 'packets' ? 'Packets' : 'Bytes'}</div>
                    <div class="text-xs text-gray-500">
                        normal p95=${fmtNum(s.normal_p95)} &middot; attack p10=${fmtNum(s.attack_p10)} p50=${fmtNum(s.attack_p50)}
                    </div>
                    <div class="text-xs ${ratioColor} font-medium">
                        separation ratio: ${s.ratio.toFixed(2)}x ${s.ok ? '(clean)' : '(WARNING: overlap — see remedies in log)'}
                    </div>
                    <div class="text-sm font-semibold">Recommended: ${fmtNum(s.recommended)}</div>
                </div>`;
        }).join('');
        $('#calibration-result').removeClass('hidden').html(`
            <div class="text-xs text-gray-500 mb-1">
                scenario: <code>${result.yaml_path}</code> &middot;
                link_bw_mbps=${result.link_bw_mbps || 'unshaped'} &middot;
                normal_bw_mbps=${result.normal_bw_mbps} &middot; n_bins=${result.n_bins}
            </div>
            ${rows}
        `);
    }

    function setCalibrationRunning(isRunning) {
        $('#calibrate-thresholds-btn').prop('disabled', isRunning);
        $('#calibrate-thresholds-btn-text').text(isRunning ? 'Calibrating…' : 'Calibrate Thresholds');
    }

    // Returns which section of env_params currently holds .thresholds.packets/bytes
    // (attacks for attack-detection scenarios, classification for the classification
    // scenario) — mirrors the scenario_env_param.yaml structure already loaded into
    // currentConfig.env_params.
    function _thresholdsSectionKey() {
        const env = currentConfig.env_params || {};
        if (env.attacks && env.attacks.thresholds) return 'attacks';
        if (env.classification && env.classification.thresholds) return 'classification';
        return null;
    }

    // Builds the "what happened" popup shown once calibration + in-memory apply settle.
    function showCalibrationModal({ title, result, appliedInMemory, errorMessage }) {
        $('#calibration-result-modal-title').text(title);
        $('#save-scenario-thresholds-modal-btn').toggleClass('hidden', !appliedInMemory);

        let html = '';
        if (result) {
            const p = result.packets, b = result.bytes;
            html += `
                <p>Measured on this server (hosts: ${result.host_names.join(', ')};
                link_bw_mbps=${result.link_bw_mbps || 'unshaped'}, normal_bw_mbps=${result.normal_bw_mbps},
                n_bins=${result.n_bins}):</p>
                <ul class="list-disc list-inside">
                    <li><b>Packets</b> — normal p95=${fmtNum(p.normal_p95)}, attack p10=${fmtNum(p.attack_p10)}
                        &rarr; recommended <b>${fmtNum(p.recommended)}</b>
                        (separation ${p.ratio.toFixed(2)}x ${p.ok ? '✓' : '⚠ overlap'})</li>
                    <li><b>Bytes</b> — normal p95=${fmtNum(b.normal_p95)}, attack p10=${fmtNum(b.attack_p10)}
                        &rarr; recommended <b>${fmtNum(b.recommended)}</b>
                        (separation ${b.ratio.toFixed(2)}x ${b.ok ? '✓' : '⚠ overlap'})</li>
                </ul>`;
            if (!p.ok || !b.ok) {
                html += `<p class="text-amber-600 text-xs">⚠ Normal and attack traffic overlap for at least one
                    metric on this server — the values below are best-effort. See the log for remedies
                    (more rounds, lower normal traffic rate, stronger attack burst, fewer bins).</p>`;
            }
        }

        if (appliedInMemory) {
            html += `<p class="text-green-700 font-medium">Applied to the loaded configuration
                      (packets=${fmtNum(result.packets.recommended)}, bytes=${fmtNum(result.bytes.recommended)})
                      — in memory only, nothing written to disk yet.</p>
                      <p class="text-xs text-gray-500">Review under Environment &rarr; Scenario if you want to
                      tweak the values further, then click "Save to scenario YAML" below (or the button on the
                      Network tab) to persist them to <code>${result.yaml_path}</code>.</p>`;
        } else if (errorMessage) {
            html += `<p class="text-red-600 font-medium">${errorMessage}</p>`;
        }

        $('#calibration-result-modal-body').html(html);
        $('#calibration-result-modal').removeClass('hidden');
    }

    // Apply (memory only): pushes the recommended values into currentConfig + the
    // rendered form inputs, then syncs to the backend's in-memory config the same
    // way the top "Save"-page's updateConfigurationInMemory() does. Never touches
    // the scenario YAML file.
    function applyCalibratedThresholdsInMemory(result) {
        const section = _thresholdsSectionKey();
        if (!section) {
            showCalibrationModal({
                title: 'Calibration Complete — could not apply',
                result,
                errorMessage: 'No attacks/classification.thresholds section found in the '
                    + 'current configuration (unexpected gym_type?). Nothing was changed.',
            });
            return;
        }

        const packets = Math.round(result.packets.recommended);
        const bytes_  = Math.round(result.bytes.recommended);

        currentConfig.env_params[section].thresholds.packets = packets;
        currentConfig.env_params[section].thresholds.bytes   = bytes_;

        $('#' + CSS.escape(`env_params.${section}.thresholds.packets`)).val(packets);
        $('#' + CSS.escape(`env_params.${section}.thresholds.bytes`)).val(bytes_);

        if (typeof updateConfigurationInMemory === 'function') {
            updateConfigurationInMemory();
        }

        showCalibrationModal({ title: 'Calibration Complete', result, appliedInMemory: true });
    }

    function saveScenarioThresholdsToDisk() {
        $.ajax({
            url: '/save_scenario_thresholds',
            type: 'POST',
            success: function (response) {
                showStatus(response.message || 'Thresholds saved.', 'success');
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error saving thresholds: ' + response.message, 'error');
            }
        });
    }

    function pollCalibrationStatus() {
        $.ajax({
            url: '/calibration_status',
            type: 'GET',
            success: function (response) {
                renderCalibrationLog(response.log);
                if (response.status === STATUS.RUNNING || response.status === STATUS.STARTING) {
                    calibrationPollTimer = setTimeout(pollCalibrationStatus, 1500);
                    return;
                }
                setCalibrationRunning(false);
                hideLoadingOverlay('calibration');
                if (response.status === STATUS.FINISHED) {
                    renderCalibrationResult(response.result);
                    applyCalibratedThresholdsInMemory(response.result);
                } else if (response.status === STATUS.ERROR) {
                    showCalibrationModal({
                        title: 'Calibration Failed',
                        errorMessage: response.error || 'Unknown error',
                    });
                }
            },
            error: function () {
                setCalibrationRunning(false);
                hideLoadingOverlay('calibration');
                showStatus('Lost contact with calibration status endpoint.', 'error');
            }
        });
    }

    $('#calibrate-thresholds-btn').click(function () {
        if ($(this).prop('disabled')) return;
        clearTimeout(calibrationPollTimer);
        $('#calibration-result').addClass('hidden');
        renderCalibrationLog(['Starting calibration…']);
        setCalibrationRunning(true);
        showLoadingOverlay('calibration', 'Calibrating thresholds on this server (builds a throwaway Mininet topology — this can take up to a minute)…');

        $.ajax({
            url: '/calibrate_thresholds',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                rounds: parseInt($('#calibration-rounds').val(), 10) || 6,
            }),
            success: function () {
                calibrationPollTimer = setTimeout(pollCalibrationStatus, 1000);
            },
            error: function (xhr) {
                setCalibrationRunning(false);
                hideLoadingOverlay('calibration');
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error starting calibration: ' + response.message, 'error');
            }
        });
    });

    $('#save-scenario-thresholds-btn').click(function () {
        saveScenarioThresholdsToDisk();
    });

    $('#save-scenario-thresholds-modal-btn').click(function () {
        saveScenarioThresholdsToDisk();
        $('#calibration-result-modal').addClass('hidden');
    });

    $('#close-calibration-result-modal-btn').click(() => {
        $('#calibration-result-modal').addClass('hidden');
    });

    // --- Modals ---
    $('#view-config-btn').click(() => {
        $('#config-modal').removeClass('hidden');
        renderConfigSummary();
    });
    $('#close-config-modal-btn').click(() => {
        $('#config-modal').addClass('hidden');
    });
    
    $('#close-model-dir-modal-btn').click(() => {
        $('#load-dir-modal').addClass('hidden');
    });


    // Load Modal Button (for future use)
    $('#load-configs-btn').click(() => {
        loadSavedConfigsTable();
        $('#config-list-modal').removeClass('hidden');
    });
    $('#refresh-saved-configs-btn').click(() => {
        loadSavedConfigsTable();
    });
    $('#load-selected-config-btn').click(() => {
        loadSelectedSavedConfig();
    });
    $('#close-modal-btn').click(() => {
        $('#config-list-modal').addClass('hidden');
    });

    $('#close-last-result-modal-btn').click(() => {
        $('#last-result-modal').addClass('hidden');
    });

    $('#mobile-nav-toggle').click(function () {
        const toast = $('#mobile-nav-toast');
        const willOpen = toast.hasClass('hidden');
        toast.toggleClass('hidden');
        $(this).attr('aria-expanded', willOpen ? 'true' : 'false');
    });

    $('#mobile-nav-toast').on('click', 'button[data-page]', function () {
        const page = $(this).data('page');
        if (page) {
            navigateTo(page);
        }
    });

    $(document).on('click', function (event) {
        const toast = $('#mobile-nav-toast');
        if (toast.hasClass('hidden')) {
            return;
        }
        const isToggle = $(event.target).closest('#mobile-nav-toggle').length > 0;
        const isInsideToast = $(event.target).closest('#mobile-nav-toast').length > 0;
        if (!isToggle && !isInsideToast) {
            toast.addClass('hidden');
            $('#mobile-nav-toggle').attr('aria-expanded', 'false');
        }
    });

});
