# ====================================================================
# AGENT ACTIONS
# Defines the discrete actions for the Agents
# ====================================================================
class AgentActions:
    ACTIONS = {
        0: "communicate normal traffic",
        1: "report attack traffic",
    }
    
    ATTACK = 1
    NORMAL_TRAFFIC = 0
    IDLE = -1    
   
    NUMBER = len(ACTIONS)

# ====================================================================
# REWARD VALUES
# Defines the reward scheme for the Host Agents
# ====================================================================
class Rewards:
    # Individual Agent Rewards
    CORRECT_NORMAL_TRAFFIC = 1.0 # reward for correct normal traffic prediction
    CORRECT_ATTACK_DETECTION = 2.0
    FALSE_POSITIVE = -1.0
    FALSE_NEGATIVE = -2.0
    
# Create instances of the classes for easy import
# These are the objects you'll import and use
AGENT_ACTIONS = AgentActions()
REWARDS = Rewards()

AGENT_STATUS_ID_MAPPING = {
    'idle': -1,
    'normal': 0,
    'attack': 1
}

HOST_STATUS_ID_MAPPING = {
    'idle': -1,
    'normal': 0,
    'under_attack': 1,
    'attacking': 2
}



