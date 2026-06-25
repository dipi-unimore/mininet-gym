
// ====================================================================\
// RESULTS PANEL RENDERING & HANDLING
// ====================================================================\

function renderResultsPanel() {

    get_results_dir_list()
        .then(list => {
            list_results_dir = list; // Store globally for sorting
            renderResultsList(list);
            syncResultsSelectionToolbar();
        })
        .catch(error => {
            console.error('Error fetching load dir list:', error);
        });


}

function get_results_dir_list() {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/get_results_dir_list',
            type: 'GET',
            success: function (response) {
                resolve(response.results_dir_list);
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                showStatus('Error fetching results dir list: ' + response.message, 'error');
                reject(response.message);
            }
        });
    });
}

// Global variable to hold the list for sorting
let list_results_dir = [];
let zoomImages = [];
let zoomIndex = -1;
let selectedResultDetail = null;
let resultsSelectionMode = false;
let selectedResultItems = new Map();
let _resultsSortState = {}; // { [scenarioId]: { col, asc } }
let _chartDescriptions = null; // cached chart_descriptions.json

function getChartDescriptions() {
    if (_chartDescriptions !== null) {
        return Promise.resolve(_chartDescriptions);
    }
    return fetch('/static/json/chart_descriptions.json')
        .then(r => r.ok ? r.json() : {})
        .then(data => { _chartDescriptions = data; return data; })
        .catch(() => { _chartDescriptions = {}; return {}; });
}

function findChartDescription(filename) {
    if (!_chartDescriptions) return null;
    const base = filename.split('/').pop().replace(/\?.*$/, '').replace(/\.png$/i, '').toLowerCase();
    // Longest keys first so "metrics_kfold" matches before "metrics"
    const keys = Object.keys(_chartDescriptions).sort((a, b) => b.length - a.length);
    for (const key of keys) {
        const k = key.toLowerCase();
        if (base === k || base.startsWith(k + '_') || base.endsWith('_' + k) || base.includes('_' + k + '_')) {
            return { _key: key, ..._chartDescriptions[key] };
        }
    }
    return { _key: base };  // no match — carry base as hint
}

function escapeAttr(value) {
    return String(value || '').replace(/'/g, "&#39;").replace(/\"/g, '&quot;');
}

function syncResultsSelectionToolbar() {
    const selectedCount = selectedResultItems.size;
    $('#results-selection-count').text(`${selectedCount} selected`);
    $('#bulk-delete-results-btn').prop('disabled', !resultsSelectionMode || selectedCount === 0);
    $('#clear-results-selection-btn').prop('disabled', selectedCount === 0);
    $('#toggle-results-selection-btn')
        .toggleClass('bg-blue-600 text-white hover:bg-blue-700', !resultsSelectionMode)
        .toggleClass('bg-slate-800 text-white hover:bg-slate-900', resultsSelectionMode)
        .text(resultsSelectionMode ? 'Exit selection mode' : 'Enable selection mode');

    $('.result-select-toggle').toggleClass('hidden', !resultsSelectionMode);
    $('.results-result-row').each(function () {
        const rowPath = String($(this).data('path') || '');
        const isSelected = resultsSelectionMode && selectedResultItems.has(rowPath);
        $(this)
            .toggleClass('bg-blue-50 ring-1 ring-blue-300', isSelected)
            .toggleClass('bg-white', !isSelected);
        $(this).find('.result-select-checkbox').prop('checked', isSelected);
    });
}

function setResultsSelectionMode(enabled) {
    resultsSelectionMode = Boolean(enabled);
    if (!resultsSelectionMode) {
        selectedResultItems.clear();
    }
    syncResultsSelectionToolbar();
}

function toggleResultSelection(path, kind = 'complete', forceSelected = null) {
    const normalizedPath = String(path || '');
    if (!normalizedPath) {
        return;
    }

    const shouldSelect = forceSelected === null
        ? !selectedResultItems.has(normalizedPath)
        : Boolean(forceSelected);

    if (shouldSelect) {
        selectedResultItems.set(normalizedPath, { kind: String(kind || 'complete') });
    } else {
        selectedResultItems.delete(normalizedPath);
    }

    syncResultsSelectionToolbar();
}

function clearResultSelection() {
    selectedResultItems.clear();
    syncResultsSelectionToolbar();
}

function deleteResultPaths(paths) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/delete_result_dirs',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ paths }),
            success: function (response) {
                resolve(response);
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                reject(response.message || 'Unable to delete selected results');
            }
        });
    });
}

function setScenarioFilterButtonState(scenarioId, mode) {
    const allButtons = $(`.scenario-filter-btn[data-scenario-id="${scenarioId}"]`);
    allButtons.removeClass('bg-blue-600 text-white border-blue-700').addClass('bg-white hover:bg-gray-100');
    const activeButton = $(`.scenario-filter-btn[data-scenario-id="${scenarioId}"][data-filter="${mode}"]`);
    activeButton.removeClass('bg-white hover:bg-gray-100').addClass('bg-blue-600 text-white border-blue-700');
}

function applyScenarioFilter(scenarioId, mode) {
    const completeSection = $(`#complete-section-${scenarioId}`);
    const incompleteSection = $(`#incomplete-section-${scenarioId}`);
    if (mode === 'complete') {
        completeSection.removeClass('hidden');
        incompleteSection.addClass('hidden');
    } else if (mode === 'incomplete') {
        completeSection.addClass('hidden');
        incompleteSection.removeClass('hidden');
    } else {
        completeSection.removeClass('hidden');
        incompleteSection.removeClass('hidden');
    }
    setScenarioFilterButtonState(scenarioId, mode);
}

// Renders the list of saved training sessions
function renderResultsList(list) {
    const dirGymTypeListEl = $('#results-list');
    dirGymTypeListEl.html('');
    const existingResultPaths = new Set();

    if (!list || list.length === 0) {
        dirGymTypeListEl.html('<p class="p-4 text-gray-500">No results found.</p>');
        return;
    }

    let tabBarHtml = '<div class="results-tab-bar flex flex-wrap gap-1 border-b border-gray-200 mb-4 pb-1">';
    let tabPanelsHtml = '';
    let firstTabId = null;

    list.forEach(gt => {
        const scenarioId = gt.gym_type.replace(/\s+/g, '-');
        if (!firstTabId) firstTabId = scenarioId;

        const completeCount = gt.data.length;
        const incompleteData = Array.isArray(gt.incomplete_data) ? gt.incomplete_data : [];
        const incompleteCount = incompleteData.length;

        tabBarHtml += `<button class="results-tab" data-results-tab="${scenarioId}">
            ${gt.gym_type} <span class="res-tab-badge">${completeCount}/${incompleteCount}</span>
        </button>`;

        let dirListDataHtml;
        let heightScroll;
        if (gt.data.length === 0) {
            heightScroll = '';
            dirListDataHtml = '<p class="p-2 text-gray-500 results-dir-item">No saved training sessions found.</p>';
        } else {
            const height = gt.data.length < 10 ? gt.data.length * 9 : 96;
            heightScroll = `h-${height} overflow-y-scroll`;
            dirListDataHtml = renderDataList(gt.gym_type, gt.data);
            gt.data.forEach(exp => existingResultPaths.add(String(exp.path || '')));
        }

        const incompleteSectionHtml = incompleteCount > 0
            ? `<div class="mt-4 border-t pt-3">
                <h4 class="text-sm font-bold text-amber-700 mb-2">Incomplete (${incompleteCount})</h4>
                <ul class="max-h-48 overflow-y-auto space-y-1">
                    ${incompleteData.map(inc => `
                        <li class="results-result-row text-xs bg-amber-50 border border-amber-200 rounded px-2 py-1 grid grid-cols-12 gap-2 items-center hover:bg-amber-100"
                            data-gym-type="${escapeAttr(gt.gym_type)}" data-path="${escapeAttr(inc.path || '')}" data-result-kind="incomplete">
                            <span class="col-span-3 font-semibold">${inc.datetime || '-'}</span>
                            <span class="col-span-4 text-gray-600 break-words" title="${inc.path || '-'}">${inc.path || '-'}</span>
                            <span class="col-span-3 text-amber-900">${inc.reason || 'Unknown reason'}</span>
                            <span class="col-span-2 text-right flex items-center justify-end gap-2">
                                <label class="result-select-toggle inline-flex items-center gap-1 text-[10px] text-gray-500 ${resultsSelectionMode ? '' : 'hidden'}">
                                    <input type="checkbox"
                                           class="result-select-checkbox h-3.5 w-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                           data-path="${escapeAttr(inc.path || '')}"
                                           data-result-kind="incomplete">
                                    <span>Select</span>
                                </label>
                                <button type="button"
                                    class="delete-incomplete-result-btn text-[11px] px-2 py-1 rounded bg-red-600 text-white hover:bg-red-700"
                                    data-path="${escapeAttr(inc.path || '')}"
                                    title="Delete incomplete result">Delete</button>
                            </span>
                        </li>
                    `).join('')}
                </ul>
            </div>`
            : `<div class="mt-4 border-t pt-3">
                <h4 class="text-sm font-bold text-amber-700 mb-2">Incomplete (0)</h4>
                <p class="text-xs text-gray-500 italic bg-gray-50 border rounded px-2 py-1">No incomplete experiments for this scenario.</p>
               </div>`;

        incompleteData.forEach(inc => existingResultPaths.add(String(inc.path || '')));

        tabPanelsHtml += `<div class="results-tab-panel hidden" id="res-tab-${scenarioId}">
            <div class="flex gap-2 mb-2">
                <button class="scenario-filter-btn px-2 py-1 text-xs rounded border bg-blue-600 text-white border-blue-700" data-scenario-id="${scenarioId}" data-filter="all">All</button>
                <button class="scenario-filter-btn px-2 py-1 text-xs rounded border bg-white hover:bg-gray-100" data-scenario-id="${scenarioId}" data-filter="complete">Complete</button>
                <button class="scenario-filter-btn px-2 py-1 text-xs rounded border bg-white hover:bg-gray-100" data-scenario-id="${scenarioId}" data-filter="incomplete">Incomplete</button>
            </div>
            <div id="complete-section-${scenarioId}">
                <div class="overflow-x-auto">
                <div class="min-w-[820px]">
                <div class="grid grid-cols-9 font-bold border-b text-xs">
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="datetime" data-label="Date Time">Date Time <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="networkconfig" data-label="Network Config">Network Config <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="trainingepisodes" data-label="Training Eps">Training Eps <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="maxsteps" data-label="Max Steps">Max Steps <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="agents" data-label="Agents">Agents <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="accuracy" data-label="Accuracy %">Accuracy % <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="testepisodes" data-label="Test Eps">Test Eps <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="result-sort-th cursor-pointer hover:text-blue-500 select-none" data-gym-type="${gt.gym_type}" data-col="score" data-label="Score %">Score % <span class="sort-ind text-gray-400">⇅</span></span>
                    <span class="text-right pr-2">Actions</span>
                </div>
                <div class="${heightScroll} gap-6">
                    <ul id="results-list-${scenarioId}" class="results-dir-list">
                        ${dirListDataHtml}
                    </ul>
                </div>
                </div>
                </div>
            </div>
            <div id="incomplete-section-${scenarioId}">
                ${incompleteSectionHtml}
            </div>
        </div>`;
    });

    tabBarHtml += '</div>';
    dirGymTypeListEl.html(tabBarHtml + tabPanelsHtml);

    const saved = (() => { try { return localStorage.getItem('activeResultsTab'); } catch (_) { return null; } })();
    const validSaved = list.find(gt => gt.gym_type.replace(/\s+/g, '-') === saved);
    switchResultsTab(validSaved ? saved : firstTabId);

    selectedResultItems = new Map(
        [...selectedResultItems.entries()].filter(([path]) => existingResultPaths.has(path))
    );
    syncResultsSelectionToolbar();
}

function switchResultsTab(tabId) {
    $('.results-tab-panel').addClass('hidden');
    $(`#res-tab-${tabId}`).removeClass('hidden');
    $('.results-tab').removeClass('cfg-tab-active');
    $(`.results-tab[data-results-tab="${tabId}"]`).addClass('cfg-tab-active');
    try { localStorage.setItem('activeResultsTab', tabId); } catch (_) {}
}

$(document).on('click', '.results-tab', function () {
    switchResultsTab($(this).data('results-tab'));
});

$(document).on('click', '.scenario-filter-btn', function () {
    const scenarioId = $(this).data('scenario-id');
    const filterMode = $(this).data('filter');
    applyScenarioFilter(scenarioId, filterMode);
});

function orderBy(gymType, col) {
    const scenarioId = gymType.replace(/\s+/g, '-');
    const s = _resultsSortState[scenarioId] || { col: null, asc: true };
    const asc = s.col === col ? !s.asc : true;
    _resultsSortState[scenarioId] = { col, asc };

    const gt = list_results_dir.find(g => g.gym_type === gymType);
    if (!gt) return;

    const sorted = [...gt.data].sort((a, b) => {
        let va, vb;
        switch (col) {
            //case 'datetime':        va = new Date(a.datetime || 0); vb = new Date(b.datetime || 0); break;
            case 'datetime':        va = String(a.datetime || ''); vb = String(b.datetime || ''); break;
            case 'networkconfig':   va = String(a.network_config || ''); vb = String(b.network_config || ''); break;
            case 'trainingepisodes':va = Number(a.training_episodes || 0); vb = Number(b.training_episodes || 0); break;
            case 'maxsteps':        va = Number(a.max_steps || 0); vb = Number(b.max_steps || 0); break;
            case 'agents':          va = a.agents_data.length; vb = b.agents_data.length; break;
            case 'accuracy':        va = Number(a.mean_accuracy || 0); vb = Number(b.mean_accuracy || 0); break;
            case 'testepisodes':    va = Number(a.test_episodes || 0); vb = Number(b.test_episodes || 0); break;
            case 'score':           va = Number(a.mean_score || 0); vb = Number(b.mean_score || 0); break;
            default: return 0;
        }
        const cmp = typeof va === 'number' && typeof vb === 'number'
            ? va - vb : (va < vb ? -1 : va > vb ? 1 : 0);
        return asc ? cmp : -cmp;
    });

    gt.data = sorted;
    $(`#results-list-${scenarioId}`).html(renderDataList(gymType, sorted));
    updateResultsSortHeaders(gymType);
}

function updateResultsSortHeaders(gymType) {
    const scenarioId = gymType.replace(/\s+/g, '-');
    const s = _resultsSortState[scenarioId] || {};
    $(`.result-sort-th[data-gym-type="${gymType}"]`).each(function () {
        const col = $(this).data('col');
        const label = $(this).data('label');
        const ind = (!s.col || s.col !== col)
            ? ' <span class="sort-ind text-gray-400">⇅</span>'
            : s.asc
                ? ' <span class="sort-ind text-blue-500">↑</span>'
                : ' <span class="sort-ind text-blue-500">↓</span>';
        $(this).html(label + ind);
    });
}

function renderDataList( gym_type, list) {
    dirListDataHtml = '';
    list.forEach(exp => {
        const isSelected = resultsSelectionMode && selectedResultItems.has(String(exp.path || ''));
        if (exp.agents_data.length >1 ) {
            agent_title = `${exp.agents_data.map(_ => _.agent_name).join(', ')}`;
            accuracy_title = `${exp.name_min_accuracy}=${exp.min_accuracy }/${exp.mean_accuracy }/${exp.name_max_accuracy}=${exp.max_accuracy }`;
            accuracy_value = `${(exp.min_accuracy * 100).toFixed(2)}/${(exp.mean_accuracy * 100).toFixed(2)}/${(exp.max_accuracy * 100).toFixed(2)}`;
            score_title = `${exp.name_min_score}=${exp.min_score }/${exp.mean_score }/${exp.name_max_score}=${exp.max_score }`;
            score_value = `${((exp.min_score / exp.test_episodes) * 100).toFixed(2)}/${(exp.mean_score / exp.test_episodes * 100).toFixed(2)}/${(exp.max_score / exp.test_episodes * 100).toFixed(2)}`;
        }
        else{
            agent_title = `${exp.agents_data[0].agent_name}`;
            accuracy_title = `${exp.mean_accuracy}`;
            accuracy_value = `${(exp.mean_accuracy * 100).toFixed(2)}`;
            score_title = `${exp.mean_score }`;
            score_value = `${((exp.mean_score / exp.test_episodes) * 100).toFixed(2)}`;

        }
        const dirItemHtml = `
            <li class="results-result-row p-2 border-b cursor-pointer hover:bg-gray-100 text-xs grid grid-cols-9 ${isSelected ? 'bg-blue-50 ring-1 ring-blue-300' : 'bg-white'}" data-gym-type="${escapeAttr(gym_type)}" data-path="${escapeAttr(exp.path)}" data-result-kind="complete">
                <span>${exp.datetime}</span>
                <span>${exp.network_config}</span>
                <span>${exp.training_episodes}</span>
                <span>${exp.max_steps}</span>
                <span title="${agent_title}">${exp.agents_data.length}</span>
                <span title="${accuracy_title}">${accuracy_value}</span>
                <span>${exp.test_episodes}</span>
                <span title="${score_title}">${score_value}</span>
                <span class="text-right flex items-center justify-end gap-2">
                    <label class="result-select-toggle inline-flex items-center gap-1 text-[10px] text-gray-500 ${resultsSelectionMode ? '' : 'hidden'}">
                        <input type="checkbox"
                               class="result-select-checkbox h-3.5 w-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                               data-path="${escapeAttr(exp.path)}"
                               data-result-kind="complete"
                               ${isSelected ? 'checked' : ''}>
                        <span>Select</span>
                    </label>
                    <button
                        type="button"
                        class="delete-result-btn text-[11px] px-2 py-1 rounded bg-red-600 text-white hover:bg-red-700"
                        data-path="${escapeAttr(exp.path)}"
                        title="Delete result">
                        Delete
                    </button>
                </span>
                <input type="hidden" value="${exp.path}">
            </li>`;
        dirListDataHtml += dirItemHtml;
    });
    return dirListDataHtml;
}

function deleteResultPath(path) {
    return new Promise((resolve, reject) => {
        $.ajax({
            url: '/delete_result_dir',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ path }),
            success: function (response) {
                resolve(response);
            },
            error: function (xhr) {
                const response = xhr.responseJSON || { message: xhr.statusText };
                reject(response.message || 'Unable to delete result');
            }
        });
    });
}

function loadResultsData(gym_type, path) {
    list_dir = list_results_dir.find(g => g.gym_type === gym_type).data;
    el = list_dir.find(e => e.path === path);
    selectedResultDetail = { gym_type, path, data: el };
    $('#result-modal').removeClass('hidden');
    renderResultModalContent(gym_type, path, el);
}

$("#close-result-panel-modal-btn").on('click', function () {
    closeResultsPanelModal();
});

function closeResultsPanelModal() {
    $('#result-modal').addClass('hidden');
    $('#result-modal-content').html('');
}

function renderResultModalContent(gym_type, path, data) {
    $('#result-path').html(path);

    // Costruiamo il percorso base per le immagini (assumendo sia relativo al path dei dati)
    const basePath = `static-training/${data.path}`;
    const testEpisodes = Number(data.test_episodes || 0);
    const trainMeanAccPct = Number(data.mean_accuracy || 0) * 100;
    const trainMinAccPct = Number(data.min_accuracy || 0) * 100;
    const trainMaxAccPct = Number(data.max_accuracy || 0) * 100;
    const testMeanScorePct = testEpisodes > 0 ? (Number(data.mean_score || 0) / testEpisodes) * 100 : 0;
    const testMinScorePct = testEpisodes > 0 ? (Number(data.min_score || 0) / testEpisodes) * 100 : 0;
    const testMaxScorePct = testEpisodes > 0 ? (Number(data.max_score || 0) / testEpisodes) * 100 : 0;
    const mitigationSummary = (data && typeof data.mitigation_summary === 'object' && data.mitigation_summary !== null)
        ? data.mitigation_summary
        : null;
    const mitigationRatioPct = mitigationSummary
        ? Number(mitigationSummary.mitigated_under_attack_ratio || 0) * 100
        : 0;
    let mitigationRatioClass = 'text-red-600';
    if (mitigationRatioPct >= 80) {
        mitigationRatioClass = 'text-green-600';
    } else if (mitigationRatioPct >= 50) {
        mitigationRatioClass = 'text-amber-600';
    }
    const agentNames = Array.isArray(data.agents_data)
        ? data.agents_data.map(a => a.agent_name).join(', ')
        : '-';
    const pngFiles = Array.isArray(data.files)
        ? data.files.filter(f => String(f).toLowerCase().endsWith('.png'))
        : [];

    // Separate analysis charts (comparison / radar / kfold) from other root PNGs
    const ANALYSIS_CHART_NAMES = ['metrics_comparison.png', 'radar_chart.png', 'metrics_kfold.png'];
    const analysisImages   = pngFiles.filter(img => ANALYSIS_CHART_NAMES.includes(String(img).toLowerCase()));
    const qtableCoverageImage = pngFiles.find(img => String(img).toLowerCase().includes('qtable_coverage')) || null;
    const otherTopImages   = pngFiles.filter(img =>
        !ANALYSIS_CHART_NAMES.includes(String(img).toLowerCase()) && img !== qtableCoverageImage
    );

    // Human-readable label for each analysis chart
    function analysisChartLabel(img) {
        const name = String(img).toLowerCase();
        if (name.includes('metrics_kfold'))      return 'K-Fold Cross-Validation';
        if (name.includes('metrics_comparison'))  return 'Metrics Comparison';
        if (name.includes('radar'))               return 'Radar Chart';
        return img;
    }

let modalContentHtml = `
        <div class="animate-fadeIn">
            <div class="flex justify-end gap-2 mb-4">
                <button id="preview-result-statuses-btn" type="button" class="bg-slate-700 hover:bg-slate-800 text-white text-sm font-semibold px-4 py-2 rounded-md shadow-sm">
                    Print statuses
                </button>
                <button id="reprint-result-charts-btn" type="button" class="bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold px-4 py-2 rounded-md shadow-sm" title="Regenerate all metric charts using the latest plot functions">
                    Reprint Charts
                </button>
                <button id="create-result-pdf-btn" type="button" class="bg-red-600 hover:bg-red-700 text-white text-sm font-semibold px-4 py-2 rounded-md shadow-sm">
                    Create PDF
                </button>
            </div>
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                <div class="bg-blue-50 p-4 rounded-xl border border-blue-100 col-span-1">
                    <h4 class="font-bold text-blue-800 mb-3 uppercase text-xs">General Stats</h4>
                    <div class="grid grid-cols-2 gap-y-3 text-sm italic">
                        <p>Train Eps: <strong>${data.training_episodes}</strong></p>
                        <p>Max Steps: <strong>${data.max_steps}</strong></p>
                        <p>Test Eps: <strong>${data.test_episodes}</strong></p>
                        <p>Mean Acc: <strong class="text-green-600">${(data.mean_accuracy * 100).toFixed(1)}%</strong></p>
                    </div>
                </div>

                <div class="lg:col-span-2 grid grid-cols-2 gap-4">
                    ${qtableCoverageImage ? `
                        <div class="border rounded-lg p-2 bg-emerald-50 border-emerald-200 col-span-2">
                            <div class="text-xs font-bold uppercase text-emerald-800 mb-2">Q-table coverage preview</div>
                            <img src="${basePath}/${qtableCoverageImage}" class="clickable-img w-full h-auto max-h-64 object-contain cursor-zoom-in rounded bg-white" alt="${qtableCoverageImage}" title="${qtableCoverageImage}" data-description="Q-table coverage preview: ${qtableCoverageImage}">
                        </div>
                    ` : ''}
                    ${otherTopImages.map(img => `
                        <div class="border rounded-lg p-1 bg-gray-50 flex flex-col gap-1">
                            <img src="${basePath}/${img}" class="clickable-img w-full h-32 object-contain cursor-zoom-in rounded" alt="${img}" title="${img}" data-description="General chart: ${img}">
                            <div class="text-[10px] text-gray-500 text-center truncate px-1" title="${img}">${img}</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            ${analysisImages.length > 0 ? `
            <div class="mb-8">
                <h3 class="text-indigo-700 font-black mb-3 uppercase text-sm flex items-center gap-2">
                    Analysis Charts
                    <span class="text-xs font-normal text-gray-400 normal-case">(metrics_comparison · radar · k-fold)</span>
                </h3>
                <div class="grid grid-cols-1 md:grid-cols-${Math.min(analysisImages.length, 3)} gap-4">
                    ${analysisImages.map(img => `
                        <div class="border rounded-xl p-2 bg-indigo-50 border-indigo-200 flex flex-col gap-1">
                            <div class="text-xs font-bold uppercase text-indigo-700">${analysisChartLabel(img)}</div>
                            <img src="${basePath}/${img}"
                                 class="clickable-img w-full h-auto max-h-64 object-contain cursor-zoom-in rounded bg-white"
                                 alt="${img}" title="${img}"
                                 data-description="Analysis chart: ${img}">
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <h3 class="text-lg font-black mb-4 flex items-center gap-2 underline decoration-blue-500">AGENTS TRAINING</h3>
            <div class="space-y-6">
                ${data.agents_data.map(agent => {
                    const agentScore = data.test_scores[agent.agent_name] || 0;
                    const cs = (data.composite_scores || {})[agent.agent_name];
                    const hasComposite = Object.keys(data.composite_scores || {}).length > 0;
                    const isWinner = hasComposite && data.name_max_winner && data.name_max_winner === agent.agent_name;
                    const isWorst  = hasComposite && data.name_min_winner && data.name_min_winner === agent.agent_name && data.name_min_winner !== data.name_max_winner;
                    const headerBg = isWinner ? 'bg-emerald-700' : isWorst ? 'bg-rose-800' : 'bg-slate-700';
                    const winnerBadge = isWinner ? '<span class="ml-2 text-yellow-300 font-black text-xs">★ WINNER</span>' : '';
                    let csBadge = '';
                    if (cs) {
                        const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';
                        csBadge = `<span class="text-[10px] opacity-80 ml-2" title="Composite winner score&#10;w_out=${(cs.w_out*100).toFixed(0)}%·recall_out=${pct(cs.recall_out)} + w_in=${(cs.w_in*100).toFixed(0)}%·recall_in=${pct(cs.recall_in)} + w_ncc=${(cs.w_ncc*100).toFixed(0)}%·recall_ncc=${pct(cs.recall_ncc)}">
                            | W-score: ${pct(cs.composite)}
                        </span>`;
                    }
                    return `
                    <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
                        <div class="${headerBg} text-white p-2 px-4 flex justify-between items-center">
                            <span class="font-bold text-sm">${agent.agent_name}${winnerBadge}</span>
                            <span class="text-[10px]">SCORE: ${agentScore}/${data.test_episodes}${csBadge}</span>
                        </div>
                        <div class="p-3 grid grid-cols-3 md:grid-cols-6 gap-2">
                            ${agent.charts.map(chart => `
                                <div class="flex flex-col gap-0.5">
                                    <img src="${basePath}/${agent.agent_name}/${chart}"
                                         class="clickable-img w-full h-auto border rounded hover:opacity-75 transition-opacity cursor-zoom-in"
                                         title="${chart}"
                                         alt="${chart}"
                                         data-description="${agent.agent_name} - ${chart}">
                                    <div class="text-[10px] text-gray-400 text-center truncate px-0.5" title="${chart}">${chart}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>`;
                }).join('')}
            </div>

            <div class="mt-10 pt-6 border-t border-gray-200">
                <h3 class="text-red-700 font-black mb-4 uppercase">Test Results</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    ${data.test_charts.map(testImg => {
                        const testImgPath = testImg.includes('/')
                            ? `${basePath}/${testImg}`
                            : `${basePath}/TEST/${testImg}`;
                        const testImgName = testImg.split('/').pop();
                        return `
                        <div class="flex flex-col gap-0.5">
                            <img src="${testImgPath}" class="clickable-img w-full border rounded shadow-sm cursor-zoom-in" alt="${testImg}" title="${testImg}" data-description="Test chart: ${testImg}">
                            <div class="text-[10px] text-gray-400 text-center truncate px-0.5" title="${testImg}">${testImgName}</div>
                        </div>
                    `;
                    }).join('')}
                </div>
            </div>

            <div class="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div class="bg-emerald-50 border border-emerald-200 rounded-xl p-4">
                    <h4 class="font-bold text-emerald-800 mb-3 uppercase text-xs">Training Summary</h4>
                    <div class="space-y-1 text-sm">
                        <p><span class="font-semibold">Scenario:</span> ${gym_type}</p>
                        <p><span class="font-semibold">Network:</span> ${data.network_config}</p>
                        <p><span class="font-semibold">Agents:</span> ${data.agents_data.length}</p>
                        <p class="text-xs text-gray-600 break-words">${agentNames}</p>
                        <p><span class="font-semibold">Episodes/Steps:</span> ${data.training_episodes} / ${data.max_steps}</p>
                        <p><span class="font-semibold">Accuracy (min/mean/max):</span> ${trainMinAccPct.toFixed(2)}% / ${trainMeanAccPct.toFixed(2)}% / ${trainMaxAccPct.toFixed(2)}%</p>
                    </div>
                </div>

                <div class="bg-amber-50 border border-amber-200 rounded-xl p-4">
                    <h4 class="font-bold text-amber-800 mb-3 uppercase text-xs">Test Summary</h4>
                    <div class="space-y-1 text-sm">
                        <p><span class="font-semibold">Test Episodes:</span> ${data.test_episodes}</p>
                        <p><span class="font-semibold">Score (min/mean/max):</span> ${data.min_score} / ${data.mean_score} / ${data.max_score}</p>
                        <p><span class="font-semibold">Score % (min/mean/max):</span> ${testMinScorePct.toFixed(2)}% / ${testMeanScorePct.toFixed(2)}% / ${testMaxScorePct.toFixed(2)}%</p>
                        <p><span class="font-semibold">Test charts:</span> ${Array.isArray(data.test_charts) ? data.test_charts.length : 0}</p>
                        ${mitigationSummary && Number(mitigationSummary.total_under_attack_count || 0) > 0 ? `
                        <p><span class="font-semibold">Mitigation episodes:</span> ${Number(mitigationSummary.episodes_with_mitigation_data || 0)}</p>
                        <p><span class="font-semibold">Under attack (total):</span> ${Number(mitigationSummary.total_under_attack_count || 0)}</p>
                        <p><span class="font-semibold">Mitigated (total):</span> ${Number(mitigationSummary.total_mitigated_under_attack_count || 0)}</p>
                        <p><span class="font-semibold">Mitigation ratio:</span> <span class="font-semibold ${mitigationRatioClass}">${mitigationRatioPct.toFixed(2)}%</span></p>
                        ` : ''}
                    </div>
                    ${(() => {
                        const cs = data.composite_scores || {};
                        const hasComposite = Object.keys(cs).length > 0;

                        if (hasComposite) {
                            // Attack scenarios: weighted composite score table
                            const agents = Object.keys(cs);
                            const pct = v => v != null ? (v * 100).toFixed(1) + '%' : '—';
                            const first = cs[agents[0]];
                            const formulaLine = `W-score = <b>${(first.w_out*100).toFixed(0)}%</b>·recall<sub>out</sub> + <b>${(first.w_in*100).toFixed(0)}%</b>·recall<sub>in</sub> + <b>${(first.w_ncc*100).toFixed(0)}%</b>·recall<sub>ncc</sub>`;
                            const rows = agents.map(name => {
                                const s = cs[name];
                                const isWinner = name === data.name_max_winner;
                                const isWorst  = name === data.name_min_winner && name !== data.name_max_winner;
                                const rowClass = isWinner ? 'bg-emerald-100 font-semibold' : isWorst ? 'bg-rose-100' : '';
                                const badge = isWinner ? ' ★' : isWorst ? ' ▼' : '';
                                return `<tr class="${rowClass}">
                                    <td class="py-1 pr-2">${escapeHtml(name)}${badge}</td>
                                    <td class="py-1 pr-2 text-center">${pct(s.composite)}</td>
                                    <td class="py-1 pr-2 text-center text-xs text-gray-500">${pct(s.recall_out)}</td>
                                    <td class="py-1 pr-2 text-center text-xs text-gray-500">${pct(s.recall_in)}</td>
                                    <td class="py-1 text-center text-xs text-gray-500">${pct(s.recall_ncc)}</td>
                                </tr>`;
                            }).join('');
                            return `
                            <div class="mt-3 pt-3 border-t border-amber-200">
                                <p class="text-xs text-amber-900 font-semibold mb-1">Winner Score <span class="font-normal text-gray-500">(inversely weighted by class frequency)</span></p>
                                <p class="text-xs text-gray-600 mb-2">${formulaLine}</p>
                                <table class="w-full text-xs">
                                    <thead><tr class="text-gray-500 border-b border-amber-200">
                                        <th class="text-left pb-1">Agent</th>
                                        <th class="pb-1">W-score</th>
                                        <th class="pb-1">recall<sub>out</sub></th>
                                        <th class="pb-1">recall<sub>in</sub></th>
                                        <th class="pb-1">recall<sub>ncc</sub></th>
                                    </tr></thead>
                                    <tbody>${rows}</tbody>
                                </table>
                            </div>`;
                        } else {
                            // Classification: simple ranking by correct predictions
                            const testScores = data.test_scores || {};
                            const eps = Number(data.test_episodes) || 1;
                            const agents = Object.entries(testScores)
                                .map(([name, score]) => ({ name, score: Number(score) || 0 }))
                                .sort((a, b) => b.score - a.score);
                            if (agents.length === 0) return '';
                            const rows = agents.map((a, i) => {
                                const isWinner = i === 0;
                                const isWorst  = i === agents.length - 1 && agents.length > 1;
                                const rowClass = isWinner ? 'bg-emerald-100 font-semibold' : isWorst ? 'bg-rose-100' : '';
                                const badge = isWinner ? ' ★' : isWorst ? ' ▼' : '';
                                const pct = ((a.score / eps) * 100).toFixed(1) + '%';
                                return `<tr class="${rowClass}">
                                    <td class="py-1 pr-2">${escapeHtml(a.name)}${badge}</td>
                                    <td class="py-1 pr-2 text-center">${a.score} / ${eps}</td>
                                    <td class="py-1 text-center">${pct}</td>
                                </tr>`;
                            }).join('');
                            return `
                            <div class="mt-3 pt-3 border-t border-amber-200">
                                <p class="text-xs text-amber-900 font-semibold mb-1">Ranking <span class="font-normal text-gray-500">(correct predictions)</span></p>
                                <table class="w-full text-xs">
                                    <thead><tr class="text-gray-500 border-b border-amber-200">
                                        <th class="text-left pb-1">Agent</th>
                                        <th class="pb-1">Correct</th>
                                        <th class="pb-1">Score %</th>
                                    </tr></thead>
                                    <tbody>${rows}</tbody>
                                </table>
                            </div>`;
                        }
                    })()}
                </div>
            </div>
        </div>`;

    $('#result-modal-content').html(modalContentHtml);
    trackContainerImagesLoading('#result-modal-content', 'result-images', 'Loading result images...');
}

function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderStatusesPreviewPopup(preview) {
    const summary = preview || {};
    const samplePayload = summary.sample_payload || { sample: [], sample_size: 0 };
    const sample = Array.isArray(samplePayload.sample) ? samplePayload.sample : [];
    const prettySample = JSON.stringify(sample, null, 2);
    const hostNames = Array.isArray(summary.host_names) ? summary.host_names : [];
    const trafficImageRelPath = String(summary.traffic_distribution_image || '');
    const trafficImageUrl = trafficImageRelPath ? encodeURI(`/static-training/${trafficImageRelPath}`) : '';

    const summaryTable = `
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-2 text-sm">
            <div class="border rounded-lg bg-white overflow-hidden">
                <div class="px-2 py-1 bg-gray-50 border-b text-xs font-semibold">Summary</div>
                <table class="w-full text-xs">
                    <tbody>
                        <tr class="border-b"><td class="p-1.5">File</td><td class="p-1.5 text-right font-medium">${escapeHtml(summary.file_name || '-')}</td></tr>
                        <tr class="border-b"><td class="p-1.5">Total entries</td><td class="p-1.5 text-right font-medium">${summary.total_entries || 0}</td></tr>
                        <tr class="border-b"><td class="p-1.5">Hosts</td><td class="p-1.5 text-right font-medium">${summary.hosts || 0}</td></tr>
                        <tr class="border-b"><td class="p-1.5">Status kinds</td><td class="p-1.5 text-right font-medium">${summary.status_kinds || 0}</td></tr>
                        <tr class="border-b"><td class="p-1.5">Attack-like entries</td><td class="p-1.5 text-right font-medium">${summary.attack_like_entries || 0}</td></tr>
                        <tr class="border-b"><td class="p-1.5">Mean packets</td><td class="p-1.5 text-right font-medium">${summary.mean_packets || 0}</td></tr>
                        <tr><td class="p-1.5">Mean bytes</td><td class="p-1.5 text-right font-medium">${summary.mean_bytes || 0}</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="border rounded-lg bg-white overflow-hidden">
                <div class="px-2 py-1 bg-gray-50 border-b text-xs font-semibold">Hosts</div>
                <div class="p-2 text-xs break-words max-h-48 overflow-y-auto">${hostNames.length ? escapeHtml(hostNames.join(', ')) : 'No host names available'}</div>
            </div>
        </div>
    `;

    const htmlContent = `
        <div class="space-y-3 text-sm">
            ${summaryTable}
            <div class="grid grid-cols-1 xl:grid-cols-2 gap-2">
                <div class="border rounded-lg bg-white overflow-hidden">
                    <div class="px-2 py-1 bg-gray-50 border-b text-xs font-semibold">Traffic Distribution (Discretized)</div>
                    <div class="p-2">
                        ${trafficImageUrl ? `
                            <img
                                src="${escapeAttr(trafficImageUrl)}"
                                alt="Discretized traffic distribution"
                                title="Discretized traffic distribution"
                                class="clickable-img w-full rounded-lg border border-gray-200 object-contain bg-white"
                            >
                        ` : `<div class="text-xs text-gray-500">No discretized traffic image available for this result.</div>`}
                    </div>
                </div>
                <div class="border rounded-lg bg-white overflow-hidden">
                    <div class="px-2 py-1 bg-gray-50 border-b text-xs font-semibold flex justify-between items-center">
                        <span>Sample (${sample.length}/${summary.total_entries || 0})</span>
                        <span class="text-gray-500">${escapeHtml(summary.file_path || '')}</span>
                    </div>
                    <pre class="max-h-[56vh] overflow-y-auto p-3 text-[11px] leading-4 whitespace-pre-wrap break-words bg-gray-900 text-gray-100">${escapeHtml(prettySample)}</pre>
                </div>
            </div>
        </div>
    `;

    return htmlContent;
}

async function downloadResultPdf(buttonEl) {
    if (!selectedResultDetail || !selectedResultDetail.data) {
        showStatus('No result selected for PDF export.', 'error');
        return;
    }

    const originalText = $(buttonEl).text();
    $(buttonEl).prop('disabled', true).text('Creating PDF...');

    try {
        const response = await fetch('/download_result_pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(selectedResultDetail)
        });

        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}`;
            try {
                const errJson = await response.json();
                errorMessage = errJson.message || errorMessage;
            } catch (e) {
                // Ignore JSON parse errors for non-JSON responses.
            }
            throw new Error(errorMessage);
        }

        const blob = await response.blob();
        const link = document.createElement('a');
        const blobUrl = window.URL.createObjectURL(blob);
        const contentDisposition = response.headers.get('Content-Disposition') || '';
        const match = contentDisposition.match(/filename=\"?([^\";]+)\"?/i);
        const filename = match && match[1] ? match[1] : 'result_report.pdf';

        link.href = blobUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(blobUrl);
        showStatus('PDF created successfully.', 'success');
    } catch (error) {
        console.error('Error creating PDF:', error);
        showStatus(`Error creating PDF: ${error.message}`, 'error');
    } finally {
        $(buttonEl).prop('disabled', false).text(originalText);
    }
}

function ensureZoomControls() {
    if ($('#zoom-controls').length > 0) {
        return;
    }
    $('#image-zoom-overlay').append(`
        <div id="zoom-controls" class="absolute inset-x-4 top-4 flex items-center justify-between pointer-events-none">
            <button id="zoom-prev-btn" class="pointer-events-auto text-white bg-black/60 hover:bg-black/80 px-3 py-2 rounded-lg text-sm font-bold" type="button">◀ Prev</button>
            <div id="zoom-caption" class="pointer-events-auto text-white bg-black/60 px-4 py-2 rounded-lg text-sm max-w-[65vw] truncate text-center"></div>
            <button id="zoom-next-btn" class="pointer-events-auto text-white bg-black/60 hover:bg-black/80 px-3 py-2 rounded-lg text-sm font-bold" type="button">Next ▶</button>
        </div>
        <div id="zoom-filename-bar" class="absolute inset-x-0 bottom-0 bg-black/70 px-4 py-2 text-center pointer-events-none">
            <div id="zoom-filename" class="text-white text-xs font-mono break-all leading-tight"></div>
        </div>
    `);
}

function updateZoomView() {
    if (zoomIndex < 0 || zoomIndex >= zoomImages.length) {
        return;
    }
    const imgEl = zoomImages[zoomIndex];
    const src = $(imgEl).attr('src');
    const caption = $(imgEl).data('description') || $(imgEl).attr('title') || $(imgEl).attr('alt') || (src ? src.split('/').pop() : 'Image');
    const filename = src ? src.split('/').pop().replace(/\?.*$/, '') : '';
    $('#zoomed-image').attr('src', src || '');
    $('#zoom-caption').text(`${zoomIndex + 1}/${zoomImages.length} - ${caption}`);
    $('#zoom-filename').text(filename);

    const $panel = $('#zoom-description-panel');
    const $title = $('#zoom-description-title');
    const $text = $('#zoom-description-text');
    $panel.addClass('hidden');

    getChartDescriptions().then(() => {
        const entry = findChartDescription(src || '');
        if (entry && entry.description) {
            $title.text(entry.title || '');
            $text.text(entry.description);
        } else {
            const hint = (entry && entry._key) ? entry._key : (src || '').split('/').pop().replace(/\?.*$/, '').replace(/\.png$/i, '');
            $title.text('');
            $text.text(`📝 No description available for this chart. Add "${hint}" to chart_descriptions.json.`);
        }
        $panel.removeClass('hidden');
    });
}

function openZoomAt(index) {
    if (!zoomImages.length) {
        return;
    }
    if (index < 0) {
        zoomIndex = zoomImages.length - 1;
    } else if (index >= zoomImages.length) {
        zoomIndex = 0;
    } else {
        zoomIndex = index;
    }
    updateZoomView();
    $('#image-zoom-overlay').fadeIn(150).removeClass('hidden').addClass('flex');
}

function closeZoomOverlay() {
    $('#image-zoom-overlay').fadeOut(150, function() {
        $(this).addClass('hidden').removeClass('flex');
    });
}

function moveZoom(step) {
    if (!zoomImages.length) {
        return;
    }
    openZoomAt(zoomIndex + step);
}

$(document).on('click', '.result-sort-th', function () {
    const gymType = $(this).data('gym-type');
    const col = $(this).data('col');
    orderBy(gymType, col);
});

$(document).ready(function() {
    ensureZoomControls();
    getChartDescriptions(); // preload so first zoom is instant
    syncResultsSelectionToolbar();

    // 1. Apertura Zoom al click su un'immagine del modal
    $(document).on('click', '.clickable-img', function() {
        const isInfoPopupImage = $(this).closest('#info-popup-text').length > 0;
        zoomImages = isInfoPopupImage
            ? $('#info-popup-text .clickable-img').toArray()
            : $('#result-modal-content .clickable-img').toArray();
        zoomIndex = zoomImages.findIndex(el => el === this);
        openZoomAt(zoomIndex);
    });

    // 2. Chiusura Zoom al click fuori dall'immagine/controlli
    $('#image-zoom-overlay').on('click', function(event) {
        if (event.target === this) {
            closeZoomOverlay();
        }
    });

    // 3. Prev/Next nel viewer
    $(document).on('click', '#zoom-prev-btn', function(event) {
        event.stopPropagation();
        moveZoom(-1);
    });

    $(document).on('click', '#zoom-next-btn', function(event) {
        event.stopPropagation();
        moveZoom(1);
    });

    $(document).on('click', '#zoomed-image, #zoom-controls', function(event) {
        event.stopPropagation();
    });

    // 4. Navigazione da tastiera quando overlay aperto
    $(document).on('keydown', function(event) {
        if (!$('#image-zoom-overlay').hasClass('flex')) {
            return;
        }
        if (event.key === 'ArrowLeft') {
            moveZoom(-1);
        } else if (event.key === 'ArrowRight') {
            moveZoom(1);
        } else if (event.key === 'Escape') {
            closeZoomOverlay();
        }
    });
    
    // 5. Gestione chiusura modal principale
    $('#close-result-panel-modal-btn').on('click', function() {
        $('#result-modal').addClass('hidden').removeClass('flex');
        closeZoomOverlay();
    });

    $(document).on('click', '#create-result-pdf-btn', function() {
        downloadResultPdf(this);
    });

    $(document).on('click', '#reprint-result-charts-btn', async function() {
        if (!selectedResultDetail || !selectedResultDetail.path) {
            showStatus('No result selected for chart reprint.', 'error');
            return;
        }

        const btn = $(this);
        const originalText = btn.text();
        btn.prop('disabled', true).text('Reprinting...');

        // Loading overlay
        const overlay = $(`
            <div id="reprint-loading-overlay"
                 style="position:fixed;inset:0;background:rgba(15,23,42,0.55);z-index:9999;display:flex;align-items:center;justify-content:center;">
                <div style="background:#fff;padding:28px 44px;border-radius:14px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,0.25);">
                    <div style="font-size:15px;font-weight:700;color:#1e293b;margin-bottom:8px;">Reprinting charts…</div>
                    <div style="font-size:12px;color:#64748b;">Regenerating metric plots with updated functions</div>
                </div>
            </div>`);
        $('body').append(overlay);

        try {
            const response = await fetch('/reprint_result_charts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gym_type: selectedResultDetail.gym_type,
                    path: selectedResultDetail.path,
                })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.message || `HTTP ${response.status}`);
            }

            const reprinted = Array.isArray(result.reprinted) ? result.reprinted.length : 0;
            const errs = Array.isArray(result.errors) && result.errors.length > 0
                ? ` (${result.errors.length} error/i: ${result.errors.join('; ')})`
                : '';

            showStatus(
                `Charts reprinted: ${reprinted} gruppo/i rigenerati.${errs}`,
                result.status === 'success' ? 'success' : 'warning'
            );

            const ts = Date.now();
            const reprList = Array.isArray(result.reprinted) ? result.reprinted : [];

            // Map reprinted tokens → PNG filenames that may have been created
            const REPRINT_FILE_MAP = {
                'metrics_kfold':                     ['metrics_kfold.png'],
                'metrics_comparison + radar_chart':  ['metrics_comparison.png', 'radar_chart.png'],
            };

            // Patch the local cache so the modal renders the new files immediately
            if (reprList.length > 0 && selectedResultDetail && selectedResultDetail.data) {
                const currentFiles = Array.isArray(selectedResultDetail.data.files)
                    ? [...selectedResultDetail.data.files]
                    : [];

                let filesChanged = false;
                for (const token of reprList) {
                    const newFiles = REPRINT_FILE_MAP[token] || [];
                    for (const fname of newFiles) {
                        if (!currentFiles.includes(fname)) {
                            currentFiles.push(fname);
                            filesChanged = true;
                        }
                    }
                }

                if (filesChanged) {
                    // Update both selectedResultDetail and the list_results_dir cache entry
                    selectedResultDetail.data.files = currentFiles;
                    const cachedEntry = list_results_dir
                        .flatMap(g => g.data || [])
                        .find(e => e.path === selectedResultDetail.path);
                    if (cachedEntry) {
                        cachedEntry.files = currentFiles;
                    }
                    // Reload modal content with updated file list
                    loadResultsData(selectedResultDetail.gym_type, selectedResultDetail.path);
                    return; // loadResultsData re-renders everything
                }
            }

            // No new files — just bust cache on existing images
            $('#result-modal-content img').each(function () {
                const src = $(this).attr('src');
                if (src) {
                    $(this).attr('src', src.split('?')[0] + '?t=' + ts);
                }
            });

        } catch (err) {
            console.error('Error reprinting charts:', err);
            showStatus(`Error reprinting charts: ${err.message}`, 'error');
        } finally {
            overlay.remove();
            btn.prop('disabled', false).text(originalText);
        }
    });

    $(document).on('click', '#preview-result-statuses-btn', async function() {
        if (!selectedResultDetail || !selectedResultDetail.path) {
            showStatus('No result selected for statuses preview.', 'error');
            return;
        }

        const originalText = $(this).text();
        $(this).prop('disabled', true).text('Loading...');

        try {
            const response = await fetch('/preview_result_statuses', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: selectedResultDetail.path, sample_size: 20 })
            });

            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}`;
                try {
                    const errJson = await response.json();
                    errorMessage = errJson.message || errorMessage;
                } catch (e) {
                    // ignore
                }
                throw new Error(errorMessage);
            }

            const preview = await response.json();
            openInfoPopupHtml(renderStatusesPreviewPopup(preview), 'Statuses distribution Preview', '/static/images/icon/info.png');
            showStatus('Statuses preview loaded successfully.', 'success');
        } catch (error) {
            console.error('Error loading statuses preview:', error);
            showStatus(`Error loading statuses preview: ${error.message}`, 'error');
        } finally {
            $(this).prop('disabled', false).text(originalText);
        }
    });

    $(document).on('click', '#toggle-results-selection-btn', function(event) {
        event.preventDefault();
        setResultsSelectionMode(!resultsSelectionMode);
    });

    $(document).on('click', '#clear-results-selection-btn', function(event) {
        event.preventDefault();
        clearResultSelection();
    });

    $(document).on('click', '#bulk-delete-results-btn', async function(event) {
        event.preventDefault();

        const selectedEntries = Array.from(selectedResultItems.entries()).map(([path, meta]) => ({ path, kind: meta.kind || 'complete' }));
        if (selectedEntries.length === 0) {
            showStatus('Select at least one result to delete.', 'error');
            return;
        }

        const paths = selectedEntries.map(entry => entry.path);
        const completeEntries = selectedEntries.filter(entry => entry.kind === 'complete');
        const incompleteCount = selectedEntries.length - completeEntries.length;
        const hasComplete = completeEntries.length > 0;

        const confirmMessage = [
            completeEntries.length === 1
                ? 'Delete this complete result folder?' 
                : `Delete these ${completeEntries.length} complete result folders?`,
            '',
            'Complete selections:',
            ...completeEntries.map(entry => `- ${entry.path}`),
            incompleteCount > 0 ? '' : null,
            incompleteCount > 0 ? `Incomplete selections: ${incompleteCount} item(s) will also be deleted.` : null,
            '',
            'This action cannot be undone.'
        ].filter(Boolean).join('\n');

        if (hasComplete && !window.confirm(confirmMessage)) {
            return;
        }

        const btn = $(this);
        const originalText = btn.text();
        btn.prop('disabled', true).text('Deleting...');

        try {
            const response = await deleteResultPaths(paths);
            const deletedCount = Array.isArray(response.deleted_paths) ? response.deleted_paths.length : 0;
            const failedCount = Array.isArray(response.failed_paths) ? response.failed_paths.length : 0;

            if (deletedCount > 0 && failedCount === 0) {
                showStatus(`Deleted ${deletedCount} result folder(s).`, 'success');
            } else if (deletedCount > 0) {
                showStatus(`Deleted ${deletedCount} result folder(s), ${failedCount} failed.`, 'warning');
            } else {
                throw new Error(response.message || 'Unable to delete selected results');
            }

            clearResultSelection();
            setResultsSelectionMode(false);
            renderResultsPanel();
        } catch (errMessage) {
            showStatus(`Error deleting selected results: ${errMessage}`, 'error');
        } finally {
            btn.prop('disabled', false).text(originalText);
        }
    });

    $(document).on('click', '.results-result-row', function(event) {
        if ($(event.target).closest('button, input, label, a').length > 0) {
            return;
        }

        const gymType = $(this).data('gym-type');
        const path = $(this).data('path');
        const kind = $(this).data('result-kind') || 'complete';

        if (resultsSelectionMode) {
            toggleResultSelection(path, kind);
            return;
        }

        loadResultsData(gymType, path);
    });

    $(document).on('click', '.result-select-checkbox', function(event) {
        event.stopPropagation();
        const rowKind = $(this).data('result-kind') || $(this).closest('.results-result-row').data('result-kind') || 'complete';
        toggleResultSelection($(this).data('path'), rowKind, $(this).is(':checked'));
    });

    $(document).on('click', '.delete-result-btn, .delete-incomplete-result-btn', async function(event) {
        event.stopPropagation();
        event.preventDefault();
        const path = $(this).data('path');
        const isIncompleteDelete = $(this).hasClass('delete-incomplete-result-btn');
        if (!path) {
            showStatus('Invalid result path.', 'error');
            return;
        }

        if (!isIncompleteDelete && !window.confirm(`Delete result folder "${path}"? This action cannot be undone.`)) {
            return;
        }

        try {
            await deleteResultPath(path);
            showStatus(`Deleted result: ${path}`, 'success');
            if (selectedResultDetail && selectedResultDetail.path === path) {
                closeResultsPanelModal();
                selectedResultDetail = null;
            }
            renderResultsPanel();
        } catch (errMessage) {
            showStatus(`Error deleting result: ${errMessage}`, 'error');
        }
    });
});
