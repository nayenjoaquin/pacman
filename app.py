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
num_episodes = 1000

values = torch.rand(num_states,num_actions)

best_reward=0
best_values=0
total_rewards=[]

for episode in range(num_episodes):
    values = torch.rand(num_states,num_actions)
    reward = run_episode(env,values)
    if(reward>best_reward):
        best_reward=reward
        best_values=values
    total_rewards.append(reward)
    print(f"episode {episode}: {reward}")

print(f"Recompensa promedio en {num_episodes} iteraciones: {sum(total_rewards)/num_episodes}")
print(f"Mejor recompensa: {best_reward}")

torch.save(best_values,"q_table.pt")

with open("best_reward.pkl","wb") as pkl_file:
    pickle.dump(best_reward,pkl_file)

eval_num_episodes=100
eval_total_rewards=[]
for episode in range(eval_num_episodes):
    reward = run_episode(env,best_values)
    eval_total_rewards.append(reward)

print(f"recompensa promedio en {eval_num_episodes} iteraciones de evaluación: {sum(eval_total_rewards)/eval_num_episodes}")





# alpha = 0.9
# gamma = 0.8

# e = 1.0
# decay = 0.1


# episodes = 100
# steps = 1000

##qtable = np.zeros((256, 9))

# for episode in range(episodes):
#     state = env.reset()
#     done = False

#     for s in range(steps):
#         if random.uniform(0, 1) < e:
#             action = env.action_space.sample()
#         else:
#             action = min(np.argmax(qtable[state, :]), 8)
#         new_state, reward, done, _ = env.step(action)

#         qtable[state, action] = qtable[state, action] + alpha * (
#             reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
#         )
#         state = new_state
#         if done:
#             break

#     e = np.exp(-decay * episode)

# print("Agente ha sido entrenado!")

# for i, q in enumerate(qtable):
#     if q[0] == 0:
#         print(i)
#         break

# Close the env
env.close()
