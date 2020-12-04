import gym
import numpy as np
from agent_util import *

def chose_action(n_act,ob,agent):
    return np.random.choice(range(n_act),p=agent.Policy[ob])
    
def train(env_name):
    #=========== Edit ========================
    GAMMA = 0.9
    ALPHA = 0.5
    N_EP = 100
    #=========== Initial ======================
    env = gym.make(env_name)

    N_ACT = env.action_space.n
    N_OB  = env.observation_space.n

    a_agent = Agent(N_OB,N_ACT,qvalue='zero')
    
    list_reward = []
    #=========== Train ======================
    for i in range(N_EP):
        ob = env.reset()
        sum_reward = 0

        while(1):
            env.render()

            act=chose_action(N_ACT,ob,a_agent)

            ob_next, reward, done, info = env.step(act)

            a_agent.QValue[ob,act] += ALPHA *(reward +  \
                                      GAMMA* max(a_agent.QValue[ob_next])\
                                      - a_agent.QValue[ob,act])
            
            ob = ob_next

            sum_reward += reward
            
            PolicyUpdate(a_agent,update='epsilon-greedy')

            if done:
                break
        list_reward.append(sum_reward)
    env.close()

if __name__=="__main__":
    env_name="FrozenLake-v0"
    train(env_name)