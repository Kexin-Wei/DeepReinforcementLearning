import gym
import pdb

env = gym.make("Breakout-v0")
ob = env.reset()

for i in range(300):
    env.render()
    act = env.action_space.sample() 

    print(i,act)

    ob, reward,done,info = env.step(act)

    pdb.set_trace()
    print(reward,done,info)

    if done:
        break

