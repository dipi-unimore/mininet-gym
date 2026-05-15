# ====================================================================
# AGENT ACTIONS
# Defines the discrete actions for the Agents
# ====================================================================
class AgentActions:
    ACTIONS = {
        0: "communicate normal traffic",
        1: "report attack traffic in",
        2: "report attack traffic out",
    }
    
    ATTACK_IN = 1
    ATTACK_OUT = 2
    NORMAL_TRAFFIC = 0
    IDLE = -1    
   
    NUMBER = len(ACTIONS)

# ====================================================================
# REWARD VALUES
# Defines the reward scheme for the Host Agents
# ====================================================================
class Rewards:
    # Individual Agent Rewards
    CORRECT_NORMAL_TRAFFIC = 0.5          # positive signal for majority class
    CORRECT_ATTACK_DETECTION = 3.0
    CORRECT_UNDER_ATTACK_DETECTION = 3.0
    LINK_OFF = -0.1
    WRONG_ATTACK_DIRECTION_DETECTED = -2.0  # worse than FP to prevent direction hedging
    FALSE_POSITIVE = -1.5                   # scaled in wrapper — symmetric with FALSE_NEGATIVE
    FALSE_NEGATIVE = -1.5                   # scaled in wrapper
    
# Create instances of the classes for easy import
# These are the objects you'll import and use
AGENT_ACTIONS = AgentActions()
REWARDS = Rewards()

AGENT_STATUS_ID_MAPPING = {
    'idle': -1,
    'normal': 0,
    'attack_in': 1,
    'attack_out': 2
}

HOST_STATUS_ID_MAPPING = {
    'idle': -1,
    'normal': 0,
    'under_attack': 1,
    'attacking': 2,
    'incoming_blocked_attack': 3,
    'out_attack_blocked': 4,
}

NORMALIZED = "normalized"
RAW = "raw"

