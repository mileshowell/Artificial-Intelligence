import gym
from numpy import array, dot, random, exp
import numpy as np
env = gym.make('CartPole-v0')
x = []
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        x.append(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
print(x)
print(x[-1])
env.close()