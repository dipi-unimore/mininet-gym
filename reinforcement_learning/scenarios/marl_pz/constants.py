COORDINATOR = "coordinator"
NORMALIZED = "normalized"
RAW = "raw"


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


class CoordinatorActions:
    ACTIONS = {
        0: "no attack",
        1: "attack detected",
    }
    NO_ATTACK = 0
    ATTACK = 1
    NUMBER = len(ACTIONS)


class Rewards:
    CORRECT_NORMAL_TRAFFIC = 0.5
    CORRECT_ATTACK_DETECTION = 3.0
    CORRECT_UNDER_ATTACK_DETECTION = 2.5
    LINK_OFF = -0.1
    WRONG_ATTACK_DIRECTION_DETECTED = -2.0
    FALSE_POSITIVE = -1.5
    FALSE_NEGATIVE = -2.5


class CoordinatorRewards:
    CORRECT_NO_ATTACK = 0.5
    CORRECT_ATTACK = 3.0
    FALSE_POSITIVE = -1.5
    FALSE_NEGATIVE = -2.5


AGENT_ACTIONS = AgentActions()
COORDINATOR_ACTIONS = CoordinatorActions()
REWARDS = Rewards()
COORDINATOR_REWARDS = CoordinatorRewards()

HOST_STATUS_ID_MAPPING = {
    'idle': -1,
    'normal': 0,
    'under_attack': 1,
    'attacking': 2,
    'incoming_blocked_attack': 3,
    'out_attack_blocked': 4,
}

COORDINATOR_STATUS_ID_MAPPING = {
    'idle': -1,
    'normal': 0,
    'attack': 1,
}
