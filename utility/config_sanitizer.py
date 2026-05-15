from utility.constants import GYM_TYPE


def clean_load_dir(config_dict, gym_types=None):
    """
    If an agent load_dir path points to a different gym_type than the
    currently selected env gym_type, disable loading for that agent.
    """
    if not isinstance(config_dict, dict):
        return config_dict

    env_params = config_dict.get("env_params", {})
    current_gym_type = env_params.get("gym_type")
    agents = config_dict.get("agents", [])

    if not current_gym_type or not isinstance(agents, list):
        return config_dict

    if gym_types is None:
        gym_types = list(GYM_TYPE.keys())

    gym_types = set(gym_types)

    for agent in agents:
        if not isinstance(agent, dict):
            continue

        load_dir = agent.get("load_dir")
        if load_dir in (None, "", "None"):
            continue

        path = str(load_dir).replace("\\", "/")
        path_parts = [part for part in path.split("/") if part]

        detected_gym_type = None
        for part in path_parts:
            if part in gym_types:
                detected_gym_type = part
                break

        if detected_gym_type and detected_gym_type != current_gym_type:
            agent["load_dir"] = None
            agent["load"] = False

    return config_dict
