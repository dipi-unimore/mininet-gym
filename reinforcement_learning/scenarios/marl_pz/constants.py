COORDINATOR = "coordinator"
NORMALIZED = "normalized"
RAW = "raw"


class CommStrategy:
    """Communication strategy identifiers for the marl_pz scenario.

    Strategies are ordered by complexity (S0 → S5):
      NONE               – no messages, no coordinator (independent agents baseline)
      NAIVE_BROADCAST    – coordinator broadcasts any attack alert to all hosts (S0)
      UAQ                – Uncertainty-Aware Querying: only confident host alerts
                           propagate; uncertain hosts are filtered out (S1)
      FEDERATED_SYNC     – periodic Q-table averaging across tabular host agents (S2)
      POLICY_EXCHANGE    – copy best-performing peer policy to lagging agents (S3)
      EXPERIENCE_SHARING – share high-TD-error transitions between tabular agents (S4)
      HIERARCHICAL       – cluster-head aggregation for large IoT topologies (S5, future)
    """
    NONE               = "none"
    NAIVE_BROADCAST    = "naive_broadcast"
    UAQ                = "uaq"
    FEDERATED_SYNC     = "federated_sync"
    POLICY_EXCHANGE    = "policy_exchange"
    EXPERIENCE_SHARING = "experience_sharing"
    HIERARCHICAL       = "hierarchical"

    ALL = [NONE, NAIVE_BROADCAST, UAQ,
           FEDERATED_SYNC, POLICY_EXCHANGE, EXPERIENCE_SHARING, HIERARCHICAL]
    IMPLEMENTED = [NONE, NAIVE_BROADCAST, UAQ,
                   FEDERATED_SYNC, POLICY_EXCHANGE, EXPERIENCE_SHARING]


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

# Column layout for the compact per-episode/per-host alert-communication array
# persisted in comm_stats.json (see marl_pz_main._build_comm_stats_payload).
ALERT_COMM_COLUMNS = ("episode", "host_idx", "total", "confident", "uncertain")

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
