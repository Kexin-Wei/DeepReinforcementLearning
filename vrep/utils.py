# %%
class Args:
    
    LR     = 1e-1
    
    EPOCHS = 50
    
    TRIAN_PER_EP = 10
    STEPS_PER_EP = 1024
    
    BATCH_SIZE = 64
    
    # parameters for td error and gae
    GAMMA  = 0.99
    LAMBDA = 0.95
    
    # clip ratio of loss_clip
    EPSILON = 0.2
    
    
    
    # parameters in loss sum calculation for value and entropy
    COEF_ENTROPY = 0.01
    COEF_VALUE   = 0.5
    
    # env steps control
    TIMESTEP = 0.05
    MAX_TIMESTEPS = 4 # max_timesteps = max_steps * timestep 
    NUM_WORKERS   = 8
    PORT = 20000
    
    
    DEVICE = None   
    fc_n = [64,64]
    
    ONE_TRAJ_FLAG = False
# %%
# # %%
import glob
import subprocess
import concurrent.futures
import vrep as v
import time
from sim_env_dynamic_single import Lapra_Sim_dynamic_single

def open_coppelia(port,hide = False):
    # check if port running : hard coded
    clientID = v.simxStart('127.0.0.1',port,True,True,5000,5)    
    if clientID!=-1: #connected
        v.simxFinish(clientID) # close this communication thread for future use
        return None      
                
    file = glob.glob("*.ttt")
    assert len(file) == 1, "There should be one and only one *.ttt file in the directory"    
    model_file = file[0]
    
    port_control = f" -gREMOTEAPISERVERSERVICE_{port}_FALSE_TRUE"#https://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
    command = f"coppeliaSim.sh {model_file}" + port_control
    
    if hide:
        # for hide mode better chose automatic quit
        command += " -h" 
        # command += " -q"# quit after simulation end
        
    #https://stackoverflow.com/questions/4996852/how-to-just-call-a-command-and-not-get-its-output
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p


def coppelia_env_make(timestep, port, hide = False, fast=False,time_episode = 0.1):    
    p = open_coppelia(port,hide=hide)            
    time.sleep(5)    
    env = Lapra_Sim_dynamic_single(timestep, port, fast=fast,time_episode = time_episode)
    return env

# %%
import platform
import os
def dir_maker(args,FILENAME):
    comment = f"lr_{args.LR}"
        
    comment += f"_ep_{args.EPOCHS}_train_{args.TRIAN_PER_EP}_steps_{args.STEPS_PER_EP}"
    for i in range(len(args.fc_n)):        
        comment += f"_fc{i+1}_{args.fc_n[i]}"
    
    
    OS = "mac" if platform.system() == "Darwin" else "linux"
    DIR = os.path.join(f"test_{OS}_{FILENAME}_workers_{args.NUM_WORKERS}",
                       comment)
    
    try:
        os.makedirs(DIR)
    except:
        print(f"Failed to open folder {DIR}")    
        
    return DIR,comment