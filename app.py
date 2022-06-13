import gym
import numpy as np
import time
import random
env = gym.make("ALE/MsPacman-v5")

# print(f"Espacio de acciones: {env.action_space}")
# print(f"Espacio de estados: {env.observation_space}")


numEpisodes = 5
numActions = 1000

for i in range(numEpisodes):
    state = env.reset()
    totalReward = 0

    for j in range(numActions):
        env.render(mode="rgb_array")

        #ACCION ALEATORIA
        randomAction = env.action_space.sample()

        observation, reward, done, info = env.step(randomAction)

        totalReward += reward
    
    print(f"Episode {i}, Total reward: {totalReward}")

env.close()



