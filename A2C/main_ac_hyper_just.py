# %%
# one step advantage actor critic
import gym
import datetime
import os
import imageio
import platform
from agent import A2C

import matplotlib.pyplot as plt
from itertools import product
from torch.utils.tensorboard import SummaryWriter

parameters = dict(
    alpha = [1e-7],
    fc1 = [2048],
    fc2 = [2048]
)
para_pairs = [v for v in parameters.values()]
RENDER_FLAG = False
TENSORBOARD_FLAG = True
EPOCHS = 3000
GAMMA = 0.99
# %%
env_name = "LunarLander-v2"
env = gym.make(env_name)

N_OB  = env.observation_space.shape[0]
N_ACT = env.action_space.n

FILENAME = os.path.splitext(os.path.basename(__file__))[0]
OS = "mac" if platform.system() == "Darwin" else "linux"
# %%
DIR = os.path.join(f"test_{OS}_{FILENAME}_{env_name}",
                   datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(DIR)
except:
    print(f"Failed to open folder {DIR}")

# %%
for alpha,fc1,fc2  in product(*para_pairs):
        A2C(alpha,GAMMA,
        env,N_OB,N_ACT,
        fc1,fc2,
        EPOCHS,DIR,
        TENSORBOARD_FLAG=TENSORBOARD_FLAG,RENDER_FLAG=RENDER_FLAG)

# %%