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