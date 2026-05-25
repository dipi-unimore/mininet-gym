// ====================================================================\
// WEB SOCKET
// ====================================================================\

function logWebSocketMessage(message, level) {
    const wsLog = $('#websocket-log');

    // Mappa le classi di colore
    const logClasses = {
        // L'intera riga, principalmente per ERRORI o SYSTEM
        LINE: {
            'ERROR': 'text-pink-400 bg-gray-800 p-1 rounded-sm',
            'SYSTEM': 'text-blue-400',
            'INFO': 'text-gray-300', // Default color for INFO
            'DEBUG': 'text-gray-300' // Default color for DEBUG
        },
        // Solo il tag [LEVEL], i colori chiari funzionano bene su sfondo scuro (bg-gray-900)
        TAG: {
            'INFO': 'text-white',      // INFO in bianco
            'DEBUG': 'text-yellow-500', // DEBUG in arancione
            'ERROR': 'text-red-500',    // ERROR in rosso
            'SYSTEM': 'text-blue-500'   // SYSTEM in blu
        }
    };

    // Normalizza il livello e applica i colori
    const upperLevel = level.toUpperCase();
    const tagClass = logClasses.TAG[upperLevel] || 'text-gray-500';
    const lineClass = logClasses.LINE[upperLevel] || 'text-gray-300';

    // Costruisce il tag e il messaggio formattati
    const formattedTag = `<span class="${tagClass}">[${upperLevel}]</span>`;
    const messageContent = message.startsWith(`[${upperLevel}]`) ? message.substring(`[${upperLevel}]`.length).trim() : message;

    const messageHtml = `<p class="${lineClass}">${formattedTag} ${messageContent}</p>`;

    wsLog.append(messageHtml);
    // Scroll down
    wsLog.scrollTop(wsLog.prop("scrollHeight"));
}

// Sync training state when socket reconnects
async function syncTrainingStateOnReconnect() {
    try {
        const response = await $.ajax({
            url: '/get_training_status',
            type: 'GET',
            timeout: 5000
        });
        
        if (response.is_training) {
            console.log('Training state synced after reconnect:', response.message);
            // Map the status response to system status
            let newState = SYSTEM_STATUS.IDLE;
            if (response.is_paused) {
                newState = SYSTEM_STATUS.PAUSED;
            } else if (response.is_stopping) {
                newState = SYSTEM_STATUS.STOPPED;
            } else if (response.status === 'RUNNING') {
                newState = SYSTEM_STATUS.TRAINING_RUNNING;
            }
            setStatus(newState, response.message);

            // Show charts and initialize for all gym types (classification/attacks have no
            // continuous data stream, so we must do this explicitly instead of relying on updateData)
            if (typeof showCharts === 'function') showCharts();
            if (currentConfig && currentConfig.cfg && currentConfig.cfg.agents && currentConfig.cfg.hosts) {
                initializeAgentsCharts(currentConfig.cfg.isMultiAgent, currentConfig.cfg.agents, currentConfig.cfg.hosts);
            }

            // Restore chart data — pass server-side agent_chart_data for page-refresh recovery
            let chartDataRestored = false;
            if (typeof restoreChartDataFromSession === 'function') {
                chartDataRestored = restoreChartDataFromSession(response.agent_chart_data);
            }

            if (chartDataRestored) {
                showStatus(`Synchronized: ${response.message}. Chart data recovered.`, 'info');
            } else {
                showStatus(`Synchronized: ${response.message}`, 'info');
            }
        } else {
            // No active training, clear status
            localStorage.removeItem('trainingStatus');
            setStatus(SYSTEM_STATUS.IDLE, 'No active training');
        }
    } catch (error) {
        console.warn('Could not sync training state on reconnect:', error);
        // Try to use localStorage as fallback
        const storedState = getTrainingStateFromStorage();
        if (storedState && storedState.status) {
            setStatus(storedState.status, 'Recovered from local state');
            if (typeof restoreChartDataFromSession === 'function') {
                restoreChartDataFromSession();
            }
        }
    }
}

function initializeWebSocket() {
    if (socket && socket.connected) {
        return; // Already connected
    }

    const wsLog = $('#websocket-log');
    wsLog.empty().append('<p class="text-blue-400">[SYSTEM] Attempting SocketIO connection...</p>');

    // NOTE: Replace with your actual SocketIO connection URL if needed
    socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

    socket.on('connect', function () {
        wsLog.append('<p class="text-green-400">[SYSTEM] SocketIO Connected.</p>');
        // On reconnect, sync the training state if we're on the training page
        if (currentPage === 'training') {
            syncTrainingStateOnReconnect();
        }
    });

    socket.on('live_update', function (messages) {
        for (const data of messages) {
            if (data.level == 'config' && data.config) {
                initializeAgentsCharts(data.config.isMultiAgent, data.config.agents, data.config.hosts); // Initialize all charts with config data
                // Keep currentConfig.cfg in sync so recovery helpers can use it
                if (typeof currentConfig !== 'undefined' && currentConfig) {
                    currentConfig.cfg = data.config;
                }
                continue;
            }
            if (data.level == 'data') {
                updateData(data); //logic to update data traffic, global state, metrics
                continue;
            }
            const message = data.message;
            if (!message) {
                continue; // Ignore empty messages
            }
            if (typeof shouldHideDropRuleMessage === 'function' && shouldHideDropRuleMessage(message)) {
                continue;
            }
            const level = data.level ? data.level.toUpperCase() : 'INFO';

            // 1. Legge lo stato delle checkbox (Devono essere aggiunte nell'HTML)
            const showInfo = $('#log-info-checkbox').prop('checked');
            const showDebug = $('#log-debug-checkbox').prop('checked');
            const showError = $('#log-error-checkbox').prop('checked');

            // 2. Decide se mostrare il messaggio in base al livello e allo stato delle checkbox
            let show = false;
            switch (level) {
                case 'INFO':
                    show = showInfo;
                    break;
                case 'DEBUG':
                    show = showDebug;
                    break;
                case 'ERROR':
                    show = showError;
                    break;
                case 'SYSTEM':
                    // I messaggi SYSTEM (es. connessione, stato training) sono sempre mostrati
                    show = true;
                    break;
                default:
                    show = showInfo; // Default a INFO se il livello non è riconosciuto
                    break;
            }

            if (show) {
                logWebSocketMessage(message, level);
            }
        }
    });

    socket.on('disconnect', function () {
        wsLog.append('<p class="text-red-400">[SYSTEM] SocketIO Disconnected.</p>');
        setStatus(SYSTEM_STATUS.DISCONNECTED, STATUS.DISCONNECTED);
        showStatus(STATUS.DISCONNECTED, 'error');
    });

    socket.on('status_update', function (data) {
        msg = `${data.mode} ${data.status}`
        if (data.status == STATUS.IDLE) {
            newState = SYSTEM_STATUS.IDLE;
        } else if (data.status === STATUS.STARTING) {
            newState = SYSTEM_STATUS.TRAINING_STARTING;
        } else if (data.status === STATUS.RUNNING) {
            if (data.mode === MODE.TRAINING) {
                newState = SYSTEM_STATUS.TRAINING_RUNNING;
            } else if (data.mode === MODE.PLOTTING) {
                newState = SYSTEM_STATUS.PLOTTING_TRAINING_DATA;
            }
            else if (data.mode === MODE.EVALUATING) {
                newState = SYSTEM_STATUS.EVALUATING_RUNNING;
            }
        } else if (data.status === STATUS.PAUSED) {
            newState = SYSTEM_STATUS.PAUSED;
        } else if (data.status === STATUS.STOPPED) {
            newState = SYSTEM_STATUS.STOPPED;
        }
        else if (data.status === STATUS.FINISHED) {
            newState = SYSTEM_STATUS.FINISHED;
        } else {
            newState = SYSTEM_STATUS.UNKNOWN; //unknown
        }
        setStatus(newState, msg);
        // Save state to localStorage for persistence across disconnections
        if (newState !== SYSTEM_STATUS.IDLE && newState !== SYSTEM_STATUS.DISCONNECTED) {
            saveTrainingStateToStorage(newState);
        }
        showStatus(`${data.message || msg}`, 'success');
    });
}