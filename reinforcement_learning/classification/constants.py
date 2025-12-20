from utility.constants import TrafficTypes
# ====================================================================
# AGENT ACTIONS
# Defines the discrete actions for the Agents
# ====================================================================
class AgentActions:
    # ACTIONS = {
    #     0: "communicate NO traffic",
    #     1: "communicate PING traffic",
    #     2: "communicate UDP traffic", 
    #     3: "communicate TCP traffic"          
    # }   
    
    ACTIONS = {
        TrafficTypes.NONE: 0,
        TrafficTypes.PING: 1,    
        TrafficTypes.UDP: 2,
        TrafficTypes.TCP: 3
    }

    
    NUMBER = len(ACTIONS)

# ====================================================================
# REWARD VALUES
# Defines the reward scheme for the Host Agents
# ====================================================================
class Rewards:
    # Individual Agent Rewards
    CORRECT_TRAFFIC = 1.0
    CLOSE = -0.80
    FARTHER = -0.90
    COMPLETELY_INCORRECT = -1.0
    
# Create instances of the classes for easy import
# These are the objects you'll import and use
AGENT_ACTIONS = AgentActions()
REWARDS = Rewards()





