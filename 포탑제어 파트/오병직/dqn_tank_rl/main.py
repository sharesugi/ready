# main.py
import torch
import numpy as np
from tank_env import TankEnv
from dqn_agent import DQNAgent
from config import Config

def train():
    env = TankEnv()
    agent = DQNAgent(state_dim=7, action_dim=10, config=Config())

    for episode in range(Config.MAX_EPISODES):
        state = env.reset()
        total_reward = 0

        for t in range(Config.MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()
            state = next_state
            total_reward += reward
            if done:
                break

        if episode % Config.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        print(f"Episode {episode}, Total reward: {total_reward:.2f}")

    torch.save(agent.policy_net.state_dict(), "dqn_tank.pth")

def test():
    env = TankEnv()
    agent = DQNAgent(state_dim=7, action_dim=10, config=Config())
    agent.policy_net.load_state_dict(torch.load("dqn_tank.pth"))
    agent.policy_net.eval()

    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        total_reward += reward

    print(f"Test episode total reward: {total_reward:.2f}")

if __name__ == "__main__":
    train()
    test()
