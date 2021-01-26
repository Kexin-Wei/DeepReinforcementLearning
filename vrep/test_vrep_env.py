# # %%
import glob
import subprocess
import signal
import concurrent.futures
import vrep as v
import os
import time
from sim_env_dynamic_single import Lapra_Sim_dynamic_single
from model import ActorCritic
from utils import coppelia_env_make
from ppo_agent import test


if __name__ == "__main__":
    start = time.perf_counter()
    
    env = coppelia_env_make(0.05,20000,time_episode = 4)
    
    INPUT_DIM  = 9
    OUTPUT_DIM = 3
    ac = ActorCritic(INPUT_DIM,OUTPUT_DIM)
    
    reward = test(0,env,ac,None)
    print(f"Reward: {reward}")
    
    
    end = time.perf_counter()
    print(f"Cost time: {end-start:.2f}")

    # os.killpg(os.getpgid(processes[-1].pid),signal.SIGTERM)
    # processes[-1].kill()
    
