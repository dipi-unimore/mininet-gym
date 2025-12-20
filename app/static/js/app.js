

// ====================================================================\
// EVENT LISTENERS & INITIALIZATION
// ====================================================================\

$(document).ready(function () {
    // Initial render of the configuration
    renderConfig();
    navigateTo('config'); // Start on the config page

    // --- Configuration Page Buttons ---
    $('#save-config-btn').click(saveConfiguration);

    $('#download-result-btn').click(downloadResults);

    // Listener for the remove agent buttons (uses event delegation)
    $('#agents-list').on('click', '.remove-agent-btn', function (e) {
        e.preventDefault();
        const agentName = $(this).data('agent-name');
        removeAgent(agentName);
    });
    $('#add-agent-btn').click(addAgent);

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
                    setStatus(SYSTEM_STATUS.TRAINING_STARTING, response.message );
                    showCharts();
                }
                else
                    setStatus(SYSTEM_STATUS.RESUMED, response.message );
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
                showStatus(response.message || 'Training session stopped.', 'success');
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error stopping training: ' + response.message, 'error');
                setStatus(SYSTEM_STATUS.ERROR, 'Error stopping training');
            }
        });
    });

    $('#view-config-btn').click(() => {
        $('#config-modal').removeClass('hidden');
        renderConfigSummary();
    });
    $('#close-config-modal-btn').click(() => {
        $('#config-modal').addClass('hidden');
    });

    // Load Modal Button (for future use)
    $('#load-configs-btn').click(() => {
        $('#config-list-modal').removeClass('hidden');
    });
    $('#close-modal-btn').click(() => {
        $('#config-list-modal').addClass('hidden');
    });

    $('#close-result-modal-btn').click(() => {
        $('#result-modal').addClass('hidden');
    });

});
