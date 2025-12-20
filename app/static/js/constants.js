// ====================================================================\
// GLOBAL CONSTANTS & VARIABLES
// ====================================================================\

//let currentConfig = JSON.parse('{{ config|tojson|safe }}'); // Load initial config from backend

// Opzioni per il menu a tendina gym_type (costante globale)
const GYM_TYPE_OPTIONS = [
    "marl_attacks",
    "marl_attacks_from_dataset",
    "classification",
    "classification_from_dataset",
    "attacks",
    "attacks_from_dataset"
];

const CLASSIFICATION = "classification";
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
    { section: 'root', key: 'net_config_filter' },
    { section: 'env_params.net_params', key: 'traffic_types' }
];

// Agent algorithm options
const ALGORITHM_OPTIONS = [
    'Q-Learning', 'SARSA', 'DQN', 'A2C', 'PPO', 'Supervised'
];

const LOG_LEVELS = [
    'info', 'debug'
];

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

