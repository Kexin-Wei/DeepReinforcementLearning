import os
import torch
import time

import concurrent.futures
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils import Args,coppelia_env_make,dir_maker
from ppo_agent import agent,train,test
from model import ActorCritic
from replay import Replay

def main():
    args = Args
    
    print("Initialing...")        
        
    # dir make
    FILENAME = os.path.splitext(os.path.basename(__file__))[0]
    DIR,comment = dir_maker(args,FILENAME)
    MODEL_FILE = f"{DIR}/Best_Model.pt"
    writer = SummaryWriter(f"{DIR}/{comment}")
    
    # Network initial
    INPUT_DIM  = 9
    OUTPUT_DIM = 3
    
    if args.NUM_WORKERS == 1:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    
    ACNet = ActorCritic(INPUT_DIM,OUTPUT_DIM,fc_n = args.fc_n,
                        device = device)
    ACNet_old = ActorCritic(INPUT_DIM,OUTPUT_DIM,fc_n = args.fc_n,
                            device = device)
    ACNet_old.load_state_dict(ACNet.state_dict())
    optimizer = torch.optim.Adam(ACNet.parameters(),lr=args.LR)
    
    
    # open environments on server side
    if args.NUM_WORKERS > 1: # open multiple coppeliasim app
        # the ports are chosed consecutively after args.PORT
        envs = []
        for i in range(args.NUM_WORKERS):
            envs.append(coppelia_env_make(args.TIMESTEP, 
                                          args.PORT + i, 
                                          time_episode = args.MAX_TIMESTEPS))
    else:
        env = open_coppelia(args.PORT)  
        envs = [env]
    
    print("Done.\nStarting Training...")
    # start train
    pbar = tqdm(range(args.EPOCHS))
    kl_max, loss_sum, reward_test = [],[],[]
    best_reward = None
    
    for ep in pbar:
        memos = Replay(BATCH_SIZE = args.BATCH_SIZE)
        
        start = time.perf_counter()
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.NUM_WORKERS)
        # agent(port,ac,args)
        results = executor.map(agent,
                               envs,
                               [ACNet]*args.NUM_WORKERS,
                               [args]*args.NUM_WORKERS)
        for result in results:
            memos.append(*result)
            
        del executor
        end = time.perf_counter()
        print(f"Collect with time: {end-start:.2f}")
        
        kl_dict,loss_dict = train(memos,ACNet,ACNet_old,optimizer,args)
        kl_max.append(sum(kl_dict["max"])/args.TRIAN_PER_EP)
        loss_sum.append(sum(loss_dict['sum']))
        
        ACNet_old.load_state_dict(ACNet.state_dict())
        
        reward = test(ep,envs[0],ACNet,args)
        reward_test.append(reward)                                   
        
        pbar_string= (f"Epoch: {ep}"
                      f"\treward: {reward:.2f}"
                      f"\tmean reward: {sum(reward_test[-100:])/min(len(reward_test),100):.2f}"
                      f"\tbest reward: {max(reward_test):.2f}")
        
        if best_reward is None or max(reward_test) > best_reward:
            best_reward = max(reward_test)
            pbar_string += f"\tmodel saved"
            torch.save(ACNet.state_dict(),MODEL_FILE)
        
        pbar.write(pbar_string)
        
        writer.add_scalar("Reward",reward,ep)
        writer.add_scalar("KL_max",kl_max[-1],ep)
        writer.add_scalar("Loss_sum_sum",loss_sum[-1],ep)
        writer.flush()
        
        
        pbar.set_postfix_str(s=f" kl:{kl_max[-1]:.3e},  loss:{loss_sum[-1]:.3f}")
    
    
    # end
    writer.close()
    
    # test
    print("Test Final Model:")
    reward_sum = 0
    for i in tqdm(range(10)):
        reward = test('Final',envs[0],ACNet,args)
        tqdm.write(f"\t {i} with reward: {reward:.2f}")
        reward_sum += reward
    print(f"Final Model Mean Reward:{reward_sum/10:.2f}")

        
    ACNet.load_state_dict(torch.load(MODEL_FILE))
    print("Test Best Model:")
    reward_sum = 0
    for i in tqdm(range(10)):
        reward = test('Best',envs[0],ACNet,args)
        tqdm.write(f"\t {i} with reward: {reward:.2f}")
        reward_sum += reward
    print(f"Best Model Mean Reward:{reward_sum/10:.2f}")
    
    # save data
    plt.figure()
    plt.plot(reward_test)
    plt.title("Reward")
    plt.savefig(f"{DIR}/reward_list_and_max_{max(reward_test):.2f}.png")
    
    plt.figure()
    plt.plot(kl_max)
    plt.title("KL Divergence")
    plt.savefig(f"{DIR}/kl_list_and_max_{max(kl_max):.2f}.png")
    
    plt.figure()
    plt.plot(loss_sum,label= 'sum')
    plt.title("Loss")
    plt.savefig(f"{DIR}/loss_list_and_sum.png")
    
if __name__ == "__main__":
    main()