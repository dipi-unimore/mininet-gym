# ====================================================================
# AGENT ACTIONS
# Defines the discrete actions for the Host Agents
# ====================================================================
class AgentActions:
    ACTIONS = {
        0: "communicate normal traffic",
        1: "dectect incoming attack traffic",
        2: "detect outgoing attack traffic",
    }
    
    IDLE = -1   
    NORMAL_TRAFFIC = 0
    INCOMING_ATTACK = 1
    OUTGOING_ATTACK = 2
    
    NUMBER = len(ACTIONS)

# ====================================================================
# COORDINATOR ACTIONS
# Defines the discrete actions for the Coordinator Agent
# ====================================================================
class CoordinatorActions:
    ACTIONS = {
        0: "do nothing",
        1: "broadcast attack alert",
        #2: "miscellaneous action" #future use
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
    CORRECT_NORMAL_TRAFFIC = 1.0 #even for coordinator
    CORRECT_ATTACK_DETECTION = 3.0
    LINK_OFF = -0.1 #the agent successfully blocked an outgoing attack, but now it taking off the link has a small negative reward
    WRONG_ATTACK_DIRECTION_DETECTED = -1.0
    FALSE_POSITIVE = -2.0
    FALSE_NEGATIVE = -3.0
    
    # Coordinator Rewards
    COORDINATOR_CORRECT_ALERT = 2.0
    COORDINATOR_FALSE_ALERT = -1.0
    COORDINATOR_MISSED_ALERT = -2.0
    
    # Shared/Team Rewards
    TEAM_SUCCESSFUL = 0 #5.0 #reward if all agents and coordinator predicted correctly

# Create instances of the classes for easy import
# These are the objects you'll import and use
AGENT_ACTIONS = AgentActions()
REWARDS = Rewards()
COORDINATOR_ACTIONS = CoordinatorActions()

HOST_STATUS_ID_MAPPING = {
    'idle': -1, #only at beginning
    'normal': 0,
    'under_attack': 1,
    'attacking': 2,
    'attacking/underattack': 3,#future use
}

COORDINATOR_LABELS = [ 'Normal', 'DOS Attack']
HOST_LABELS = [ 'Normal', 'DOS IN', 'DOS OUT']

COORDINATOR_STATUS_ID_MAPPING = {
    'idle': -1, #only at beginning
    'normal': 0,
    'attack': 1,
}

COORDINATOR = "coordinator"


