import random
import gym
import time
import numpy as np
import torch
import pickle

def run_episode(env, values):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state,values)).item()
        state,reward,is_done,info = env.step(action)
        total_reward+=reward
    return total_reward

env = gym.make("ALE/MsPacman-v5", obs_type="ram", full_action_space=False, render_mode="human")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_episodes = 1000
values = torch.load("q_table.pt")

for episode in range(num_episodes):
    reward = run_episode(env,values)
    print(f"episode {episode}: {reward}")