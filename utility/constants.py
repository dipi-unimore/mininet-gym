CLASSIFICATION_FROM_DATASET = 'classification_from_dataset'
CLASSIFICATION = 'classification'

ATTACKS = 'attacks'
ATTACKS_FROM_DATASET = 'attacks_from_dataset'

MARL_ATTACKS = 'marl_attacks'
MARL_ATTACKS_FROM_DATASET = 'marl_attacks_from_dataset'   
FROM_DATASET = 'from_dataset'

GYM_TYPE = {
    CLASSIFICATION_FROM_DATASET: 0, 
    CLASSIFICATION: 1,
    ATTACKS: 4,
    ATTACKS_FROM_DATASET: 5,
    MARL_ATTACKS: 6,
    MARL_ATTACKS_FROM_DATASET: 7
}

Q_LEARNING = 'q-learning'
DQN = 'dqn'
SARSA = 'sarsa'
A2C = 'a2c'
PPO = 'ppo'
SUPERVISED = 'supervised'

ALGORITHMS = {
    Q_LEARNING: 0,
    SARSA: 1,
    A2C: 2,
    PPO: 3,
    SUPERVISED: 4
}


NORMAL = "normal"
ATTACK = "attack"
SHORT_ATTACK = "short_attack"
LONG_ATTACK = "long_attack"


class SystemLevels:
    INFO = "info"
    DEBUG = "debug"
    ERROR = "error"
    STATUS = "status"
    CONFIG = "config"
    DATA = "data"
    
SYSTEM = "system"
    
class SystemStatus:
    IDLE= "idle"
    STARTING= "starting"
    PAUSED= "paused"
    RESUMED= "resumed"
    RUNNING= "running"
    STOPPED= "stopped"
    FINISHED= "finished"
    DISCONNECTED="disconnected"
    ERROR= "error"
    
class SystemModes: 
    TRAINING = "training"
    PLOTTING = "plotting"
    EVALUATION = "evaluating"
    
class HostStatus:   
    ATTACKING = "attacking"
    UNDER_ATTACK = "under_attack"
    WAR = "under_attack/attacking"

class TrafficTypes:     
    PING = "ping"
    UDP = "udp"
    TCP = "tcp"
    NONE = "none"
    
TRAFFIC_TYPE_ID_MAPPING = {
    TrafficTypes.NONE: 0,
    TrafficTypes.PING: 1,
    TrafficTypes.UDP: 2,
    TrafficTypes.TCP: 3,
}


