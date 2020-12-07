import gym
import matplotlib.pyplot as plt
import os

DIR = "gym_graph"

env = gym.make('Breakout-v0')
env.reset()
img = plt.imshow(env.render(mode='rgb_array')) # only call this once
print(env.render(mode='rgb_array').shape)

for i in range(10):
    plt.imsave(os.path.join(DIR,str(i)+'.png'),env.render(mode='rgb_array'))
    env.step(env.action_space.sample())
env.close()