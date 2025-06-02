# config.py
class Config:
    HOST = '0.0.0.0'
    PORT = 5000
    GAMMA = 0.99
    LR = 0.001
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 500
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE = 10
    MAX_EPISODES = 500
    MAX_STEPS = 200
