$(document).ready(function () {
  const defaultAgent = {
    name: "",
    algorithm: "",
    learning_rate: 0.01,
    discount_factor: 0.99
  };

  function addAgentForm(agent = defaultAgent) {
    const index = $("#agents-config .agent-block").length;
    const html = `
      <div class="agent-block border p-3 mb-2 rounded bg-light">
        <h6>Agent ${index + 1}</h6>
        <input type="text" class="form-control mb-1" name="agent_name" placeholder="Name" value="${agent.name}">
        <input type="text" class="form-control mb-1" name="algorithm" placeholder="Algorithm" value="${agent.algorithm}">
        <input type="number" step="0.001" class="form-control mb-1" name="learning_rate" placeholder="Learning Rate" value="${agent.learning_rate}">
        <input type="number" step="0.01" class="form-control mb-1" name="discount_factor" placeholder="Discount Factor" value="${agent.discount_factor}">
      </div>`;
    $("#agents-config").append(html);
  }

  // Load config from API
  function loadConfig() {
    $.getJSON("/params", function (data) {
    // Load general config (root-level keys excluding env and agents)
      let generalHTML = '<h5>General Configuration</h5>';
      for (const [key, value] of Object.entries(data)) {
        if (key !== "env_params" && key !== "agents") {
          generalHTML += `
            <div class="mb-2">
              <label class="form-label">${key}</label>
              <input type="text" class="form-control" name="${key}" value="${value}">
            </div>`;
        }
      }
      $("#general-config").html(generalHTML);


      // Load env config
      let envHTML = '<h5>Environment Configuration</h5>';
      for (const [key, value] of Object.entries(data.env_params)) {
        if (typeof value === 'object') {
          envHTML += `<div class="border p-2 mb-2 bg-white"><strong>${key}</strong>`;
          for (const [subkey, subvalue] of Object.entries(value)) {
            envHTML += `
              <div class="mb-2">
                <label class="form-label">${subkey}</label>
                <input type="text" class="form-control" name="${key}.${subkey}" value="${subvalue}">
              </div>`;
          }
          envHTML += `</div>`;
        } else {
          envHTML += `
            <div class="mb-2">
              <label class="form-label">${key}</label>
              <input type="text" class="form-control" name="env.${key}" value="${value}">
            </div>`;
        }
      }
      $("#env-config").html(envHTML);

      // Load agents
      $("#agents-config").empty().append('<h5>Agents Configuration</h5>');
      data.agents.forEach(agent => addAgentForm(agent));
    });
  }

  // On page load
  loadConfig();

  // Add new agent
  $("#addAgent").click(function () {
    addAgentForm();
  });

  // Placeholder for training start
  $("#start-training").click(function () {
    $("#training-output").append("<p><b>Training started...</b></p>");
    // TODO: Add WebSocket logic
  });

  // Save Configuration (TO DO)
  $("#config-form").submit(function (e) {
    e.preventDefault();
    alert("Saving configuration not implemented yet.");
  });
});
