
from A2C.main_ac import EPOCHS

import gym
import multiprocessing

from model import FC
class Args:
    EPOCHS = 3000
    
    env_name = "LunarLanderContinuous-v2"
    

def main():
    args = Args
    
    env = gym.make(args.env_name)
    N_OB = env.observation_space.shape[0]
    N_ACT = env.action_space.shape[0]
    
    actor  = FC(N_OB,N_ACT)
    critic = FC(N_OB,N_ACT)
    for i in range(args.EPOCHS):
if __name__ == "__main__":
    main()