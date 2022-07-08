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


env = gym.make("ALE/MsPacman-v5", obs_type="ram", full_action_space=False)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

values = torch.rand(num_states,num_actions)

with open("best_reward.pkl", 'rb') as pickle_file:
    best_reward = pickle.load(pickle_file)
print(f"recompensa a batir: {best_reward}")
best_values=0
better_values=False

while not better_values:
    values = torch.rand(num_states,num_actions)
    reward = run_episode(env,values)
    if(reward>best_reward):
        best_reward=reward
        best_values=values
    print(f"recompensa obtenida: {reward}")

torch.save(best_values,"q_table.pt")
with open("best_reward.pkl","wb") as pkl_file:
    pickle.dump(best_reward,pkl_file)

eval_num_episodes=100
eval_total_rewards=[]
for episode in range(eval_num_episodes):
    reward = run_episode(env,best_values)
    eval_total_rewards.append(reward)

print(f"recompensa promedio en {eval_num_episodes} iteraciones de evaluación: {sum(eval_total_rewards)/eval_num_episodes}")
