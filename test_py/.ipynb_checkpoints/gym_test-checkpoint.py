import gym
env = gym.make("FrozenLake-v0")
env.reset()
for _ in range(1000):
    env.render()
    action=int(input("0-3"))
    env.step(action)
    #env.step(env.action_space.sample()) # take a random action
env.close()