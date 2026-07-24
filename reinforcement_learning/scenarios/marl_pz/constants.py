COORDINATOR = "coordinator"
NORMALIZED = "normalized"
RAW = "raw"


class CommStrategy:
    """Communication strategy identifiers for the marl_pz scenario.

    Per paper 7 (§Coordination Strategies), these are not one flat list — they
    split into two independent tiers:

      Tier A — alerting channel (mutually exclusive; single active value).
        Each option defines an incompatible encoding for the same
        host→coordinator message channel, so exactly one can be active:
          NONE               – no messages, no coordinator (independent agents baseline)
          NAIVE_BROADCAST    – coordinator broadcasts any attack alert to all hosts (S0)
          UAQ                – Uncertainty-Aware Querying: only confident host alerts
                               propagate; uncertain hosts are filtered out (S1)

      Tier B — peer coordination mechanisms (combinable; zero or more active
      at once). These act directly on host Q-tables/policy weights,
      independent of the coordinator channel:
          FEDERATED_SYNC     – periodic Q-table averaging across tabular host agents (S2)
          POLICY_EXCHANGE    – copy best-performing peer policy to lagging agents (S3)
          EXPERIENCE_SHARING – share high-TD-error transitions between tabular agents (S4)

    HIERARCHICAL is kept only as an inert placeholder: per paper 7
    (§Communication Architecture), peer-to-peer vs. hierarchical is an
    orthogonal *architecture* axis applying to any Tier B mechanism, not a
    strategy of its own — it is intentionally excluded from IMPLEMENTED and
    from TIER_A/TIER_B, and is not selectable from the UI/config.
    """
    NONE               = "none"
    NAIVE_BROADCAST    = "naive_broadcast"
    UAQ                = "uaq"
    FEDERATED_SYNC     = "federated_sync"
    POLICY_EXCHANGE    = "policy_exchange"
    EXPERIENCE_SHARING = "experience_sharing"
    HIERARCHICAL       = "hierarchical"

    TIER_A = [NONE, NAIVE_BROADCAST, UAQ]
    TIER_B = [FEDERATED_SYNC, POLICY_EXCHANGE, EXPERIENCE_SHARING]

    ALL = [NONE, NAIVE_BROADCAST, UAQ,
           FEDERATED_SYNC, POLICY_EXCHANGE, EXPERIENCE_SHARING, HIERARCHICAL]
    IMPLEMENTED = [NONE, NAIVE_BROADCAST, UAQ,
                   FEDERATED_SYNC, POLICY_EXCHANGE, EXPERIENCE_SHARING]

    # Tier B combinations flagged as risky in docs/marl_pz_communication.md's
    # "Combining strategies" table — surfaced as warnings (not hard blocks) in
    # both training logs (marl_pz_main._train_marl_agent) and the config UI.
    TIER_B_WARNINGS = [
        (
            (FEDERATED_SYNC, POLICY_EXCHANGE),
            "S2+S3: redundancy risk — S3's periodic best-peer copy can undo "
            "the diversity S2's averaging relies on. Prefer running one or "
            "the other unless you are deliberately measuring the interaction.",
        ),
        (
            (POLICY_EXCHANGE, EXPERIENCE_SHARING),
            "S3+S4: overwrite conflict — S3 overwrites a lagging agent's "
            "Q-table/weights wholesale, while S4 updates them incrementally; "
            "S4 updates applied earlier in the same step/episode are lost "
            "when S3 fires.",
        ),
    ]


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
    CORRECT_UNDER_ATTACK_DETECTION = 3.0
    LINK_OFF = -0.1
    WRONG_ATTACK_DIRECTION_DETECTED = -2.0
    FALSE_POSITIVE = -1.5
    FALSE_NEGATIVE = -2.0


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
