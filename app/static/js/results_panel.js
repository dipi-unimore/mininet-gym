
// ====================================================================\
// RESULTS PANEL RENDERING & HANDLING
// ====================================================================\

function renderResultsPanel() {

    get_results_dir_list()
        .then(list => {
            list_results_dir = list; // Store globally for sorting
            renderResultsList(list);
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

// Renders the list of saved training sessions
function renderResultsList(list) {
    const dirGymTypeListEl = $('#results-list');
    list.forEach(gt => {
        if (gt.data.length === 0) {
            heightScroll = '';
            dirListDataHtml = '<p class="p-2 text-gray-500  results-dir-item ">No saved training sessions found.</p>';
        }
        else {
            height = gt.data.length < 10 ? gt.data.length * 9 : 96;
            heightScroll = `h-${height} overflow-y-scroll`;
            dirListDataHtml = renderDataList(gt.gym_type,gt.data);
        }
        const dirGymTypeHtml = `
           <details class="mb-4 border border-gray-300 rounded-lg">
               <summary class="bg-gray-200 px-4 py-2 rounded-t-lg cursor-pointer font-semibold text-gray-800 hover:bg-gray-300">
                   <img src="static/images/gif/earth.gif" alt="Gym Type" title="Gym Type" class="inline-block w-6 h-6 ml-2">
                   ${gt.gym_type}
               </summary>
               <div class="px-4 py-2">
                    <div class="grid grid-cols-8 font-bold border-b">
                        <span id="sort-datetime-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Date Time</span>
                        <span id="sort-networkconfig-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Network Config</span>
                        <span id="sort-trainingepisodes-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Training Episodes</span>
                        <span id="sort-maxsteps-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Max Steps</span>
                        <span id="sort-agents-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Agents</span>
                        <span id="sort-accuracy-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Accuracy %</span>
                        <span id="sort-testepisodes-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Test Episodes</span>
                        <span id="sort-score-${gt.gym_type.replace(/\s+/g, '-')}" class="cursor-pointer hover:text-blue-500">Score (Min, Mean, Max) %</span>
                    </div>

                    <div class="${heightScroll} gap-6">               
                        <ul id="results-list-${gt.gym_type.replace(/\s+/g, '-')}" class="results-dir-list">
                            ${dirListDataHtml}
                        </ul>
                    </div>
               </div>
           </details>`;

        // ✅ Sorting handlers
        $(document).on('click', `#sort-datetime-${gt.gym_type.replace(/\s+/g, '-')}`, function () {
            orderBy($(this), 'datetime');
        });

        $(document).on('click', `#sort-accuracy-${gt.gym_type.replace(/\s+/g, '-')}`, function () {
            orderBy($(this), 'accuracy');
        });

        $(document).on('click', `#sort-score-${gt.gym_type.replace(/\s+/g, '-')}`, function () {
            orderBy($(this), 'score');
        });

        dirGymTypeListEl.append(dirGymTypeHtml);
    });

}

function orderBy(el, criteria) {
    gym_type = el[0].id.split('-')[2];
    list_dir = list_results_dir.find(g => g.gym_type === gym_type).data;
    if (criteria === 'datetime')
        list_dir.sort((a, b) => new Date(b.datetime) - new Date(a.datetime)); // Newest first
    else if (criteria === 'accuracy')
        list_dir.sort((a, b) => b.mean_accuracy - a.mean_accuracy); // Highest accuracy first
    else if (criteria === 'score')
        list_dir.sort((a, b) => b.mean_score - a.mean_score);
    html = renderDataList(list_dir);
    $(`#results-list-${gym_type}`).html(html);
}

function renderDataList( gym_type, list) {
    dirListDataHtml = '';
    list.forEach(exp => {
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
            <li class="p-2 border-b cursor-pointer hover:bg-gray-100 results-dir-item grid grid-cols-8" onclick="loadResultsData('${gym_type}', '${exp.path}')">
                <span>${exp.datetime}</span>
                <span>${exp.network_config}</span>
                <span>${exp.training_episodes}</span>
                <span>${exp.max_steps}</span>
                <span title="${agent_title}">${exp.agents_data.length}</span>
                <span title="${accuracy_title}">${accuracy_value}</span>
                <span>${exp.test_episodes}</span>
                <span title="${score_title}">${score_value}</span>
                <input type="hidden" value="${exp.path}">
            </li>`;
        dirListDataHtml += dirItemHtml;
    });
    return dirListDataHtml;
}

function loadResultsData(gym_type, path) {
    list_dir = list_results_dir.find(g => g.gym_type === gym_type).data;
    el = list_dir.find(e => e.path === path);
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

let modalContentHtml = `
        <div class="animate-fadeIn">
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
                    ${data.files.filter(f => f.endsWith('.png')).map(img => `
                        <div class="border rounded-lg p-1 bg-gray-50">
                            <img src="${basePath}/${img}" class="clickable-img w-full h-32 object-contain cursor-zoom-in rounded" alt="${img}">
                        </div>
                    `).join('')}
                </div>
            </div>

            <h3 class="text-lg font-black mb-4 flex items-center gap-2 underline decoration-blue-500">AGENTS TRAINING</h3>
            <div class="space-y-6">
                ${data.agents_data.map(agent => {
                    const agentScore = data.test_scores[agent.agent_name] || 0;
                    return `
                    <div class="bg-white border rounded-xl overflow-hidden shadow-sm">
                        <div class="bg-slate-700 text-white p-2 px-4 flex justify-between items-center">
                            <span class="font-bold text-sm">${agent.agent_name}</span>
                            <span class="text-[10px]">SCORE: ${agentScore}/${data.test_episodes}</span>
                        </div>
                        <div class="p-3 grid grid-cols-3 md:grid-cols-6 gap-2">
                            ${agent.charts.map(chart => `
                                <img src="${basePath}/${agent.agent_name}/${chart}" 
                                     class="clickable-img w-full h-auto border rounded hover:opacity-75 transition-opacity cursor-zoom-in" 
                                     title="${chart}">
                            `).join('')}
                        </div>
                    </div>`;
                }).join('')}
            </div>

            <div class="mt-10 pt-6 border-t border-gray-200">
                <h3 class="text-red-700 font-black mb-4 uppercase">Test Results</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    ${data.test_charts.map(testImg => `
                        <img src="${basePath}/TEST/${testImg}" class="clickable-img w-full border rounded shadow-sm cursor-zoom-in">
                    `).join('')}
                </div>
            </div>
        </div>`;

    $('#result-modal-content').html(modalContentHtml);
}

$(document).ready(function() {
    // 1. Apertura Zoom al click su un'immagine del modal
    $(document).on('click', '.clickable-img', function() {
        const src = $(this).attr('src');
        $('#zoomed-image').attr('src', src);
        $('#image-zoom-overlay').fadeIn(200).removeClass('hidden').addClass('flex');
    });

    // 2. Chiusura Zoom al click sull'overlay
    $('#image-zoom-overlay').on('click', function() {
        $(this).fadeOut(200, function() {
            $(this).addClass('hidden').removeClass('flex');
        });
    });
    
    // 3. Gestione chiusura modal principale (il tuo codice esistente)
    $('#close-result-panel-modal-btn').on('click', function() {
        $('#result-modal').addClass('hidden').removeClass('flex');
    });
});