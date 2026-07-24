// ====================================================================\
// GLOBAL CONSTANTS & VARIABLES
// ====================================================================\

//let currentConfig = JSON.parse('{{ config|tojson|safe }}'); // Load initial config from backend

// Opzioni per il menu a tendina gym_type (costante globale)
const GYM_TYPE_OPTIONS = [
    "marl_pz",
    "marl_pz_from_dataset",
    "marl_attacks",
    "marl_attacks_from_dataset",
    "classification",
    "classification_from_dataset",
    "attacks",
    "attacks_from_dataset",
    "attacks_ho",
    "attacks_ho_from_dataset"
];

const CLASSIFICATION = "classification";
const SINGLE_AGENT_HOST_OBSERVABLE = "attacks_ho";
const MARL = "marl";

const COORDINATOR = "coordinator";

// Keys that are in the root of the config object (conceptually 'general')
const ROOT_KEYS = Object.keys(currentConfig).filter(key => {
    const value = currentConfig[key];
    return typeof value !== 'object' || value === null;  // || Array.isArray(value);
});
// List of fields that MUST remain read-only
const READ_ONLY_FIELDS = [
    { section: 'root', key: 'training_directory' },
    { section: 'root', key: 'enable_web_interface' },
    { section: 'root', key: 'web_server_port' },
    { section: 'root', key: 'server_user' },
    { section: 'root', key: 'net_config_filter' },
    { section: 'root', key: 'training_execution_directory' },
    { section: 'env_params.net_params', key: 'traffic_types' }
];

// List of fields that MUST remain read-only
const HIDDEN_FIELDS = [
    { section: 'root', key: 'training_directory' },
    { section: 'root', key: 'enable_web_interface' },
    { section: 'root', key: 'web_server_port' },
    { section: 'root', key: 'server_user' },
    { section: 'root', key: 'net_config_filter' },
    { section: 'root', key: 'training_execution_directory' },
];

// Agent algorithm options
const ALGORITHM_OPTIONS = [
    'Q-Learning', 'SARSA', 'DQN', 'A2C', 'PPO', 'Supervised'
];

const LOG_LEVELS = [
    'info', 'debug'
];

const STATE_INPUT_MODES = [
    'normalized', 'raw'
];

// Tier A — alerting channel (mutually exclusive: single-select). Each option
// defines an incompatible encoding for the same host→coordinator message
// channel, so exactly one can be active. 'hierarchical' is intentionally
// absent here — per paper 7 it's an orthogonal, not-yet-implemented
// communication *architecture* axis, not a strategy (see constants.py
// CommStrategy docstring / docs/marl_pz_communication.md).
const COMM_STRATEGY_TIER_A_OPTIONS = ['none', 'naive_broadcast', 'uaq'];

// Tier B — peer coordination mechanisms (combinable: multi-select). These act
// on host Q-tables/policy weights directly, independent of the Tier A
// coordinator channel.
const COMM_STRATEGY_TIER_B_OPTIONS = ['federated_sync', 'policy_exchange', 'experience_sharing'];

// Tier B combinations flagged as risky (not blocked) — mirrors
// CommStrategy.TIER_B_WARNINGS in constants.py / the "Combining strategies"
// table in docs/marl_pz_communication.md.
const COMM_STRATEGY_TIER_B_WARNINGS = [
    {
        pair: ['federated_sync', 'policy_exchange'],
        message: "S2+S3: redundancy risk — S3's periodic best-peer copy can undo the diversity S2's averaging relies on. Prefer running one or the other unless you are deliberately measuring the interaction.",
    },
    {
        pair: ['policy_exchange', 'experience_sharing'],
        message: "S3+S4: overwrite conflict — S3 overwrites a lagging agent's Q-table/weights wholesale, while S4 updates them incrementally; S4 updates applied earlier in the same step/episode are lost when S3 fires.",
    },
];

let algoDefaults = {}; // keyed by lowercase algorithm name, loaded from /get_algo_defaults

let socket = null;
let systemStatus = 0; //o idle, 1 stopped/to start, 2 paused, 3 training, negative value to indicate action refused, but never set
let currentPage = 'config'; // Initial page


const STATUS = {
    IDLE: "idle",
    STOPPED: "stopped",
    PAUSED: "paused",
    STARTING: "starting",
    RUNNING: "running",
    RESUMED: "resumed",
    FINISHED: "finished",
    DISCONNECTED: "disconnected",
    ERROR: "error"
};

const SYSTEM_STATUS = {
    IDLE: 0,
    STOPPED: 1,
    PAUSED: 2,
    TRAINING_STARTING: 3,
    TRAINING_RUNNING: 4,
    PLOTTING_TRAINING_DATA: 5,
    EVALUATING_RUNNING: 6,
    RESUMED: 7,
    FINISHED: 8,
    DISCONNECTED: -1,
    ERROR: -2,
    UNKNOWN: -3
};

const MODE = {
    TRAINING: "training",
    PLOTTING: "plotting",
    EVALUATING: "evaluating"
};

