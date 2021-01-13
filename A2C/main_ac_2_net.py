# %%
# one step advantage actor critic
import gym
import datetime
import os
import platform
from agent import A2C_2_NET
from dm_wrapper import make_env



# %%
env_name = "PongDeterministic-v4"
env_name = "LunarLander-v2"
if 'NoFrameskip' in env_name:
    CNN_NET_FLAG = True
    env = make_env(env_name)
    N_OB  = env.observation_space.shape
    N_ACT = env.action_space.n
    
    lr = 1e-5
    COEF_ENT   = 0.1
    COEF_VALUE = 0.5
    GAMMA = 0.99
    fc1=2048
    fc2=2048
    RENDER_FLAG      = True
    TENSORBOARD_FLAG = True
    EPOCHS = 40_000/4
    
else:
    CNN_NET_FLAG = False
    env = gym.make(env_name)
    N_OB  = env.observation_space.shape[0]
    N_ACT = env.action_space.n
    lr = 1e-6
    COEF_ENT   = 0.01
    COEF_VALUE = 0.5
    GAMMA = 0.99
    fc1=1024
    fc2=1024
    RENDER_FLAG      = False
    TENSORBOARD_FLAG = True
    EPOCHS = 2000



# %%
FILENAME = os.path.splitext(os.path.basename(__file__))[0]
OS = "mac" if platform.system() == "Darwin" else "linux"
DIR = os.path.join(f"test_{OS}_{FILENAME}_Entropy_{env_name}",
                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    print(f"Failed to open folder {DIR}")

# %%

A2C_2_NET(lr,COEF_ENT,COEF_VALUE,GAMMA,
          env,N_OB,N_ACT,
          fc1,fc2,
          EPOCHS,DIR,
          TENSORBOARD_FLAG=TENSORBOARD_FLAG,
          RENDER_FLAG=RENDER_FLAG)

# %%