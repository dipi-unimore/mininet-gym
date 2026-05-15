tailwind.config = {
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
        }
    }
}

function isClassificationEnv() {
    return currentConfig.env_params.gym_type.startsWith(CLASSIFICATION);
}

function isSingleAgentHostObservableEnv(){
    return currentConfig.env_params.gym_type.startsWith(SINGLE_AGENT_HOST_OBSERVABLE);
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