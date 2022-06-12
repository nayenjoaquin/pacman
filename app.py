import gym
import time
env = gym.make("ALE/MsPacman-v5", render_mode="human")

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



