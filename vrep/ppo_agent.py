from sim_env_dynamic_single import Lapra_Sim_dynamic_single
from replay import Replay
import torch
# %%
def one_trajectory(env,ac,len_old,len_target,args):    
    
    obs  = []
    obs_ = []
    acts = []
    dones = []
    rewards = []
    advantages = []        
    
    step = 0
    env.reset_sim()
    ob = env.get_states()
    
    while 1:        
        act = ac.get_act(ob)
        
        # env.step()
        env.action_update(act)
        reward, done = env.get_rewards()        
        ob_ = env.get_states()
        
        ob = ob_
                
        step += 1
        
        obs.append(ob)
        obs_.append(ob_)
        acts.append(act)
        dones.append(done)
        rewards.append(reward)
        
        if step+len_old >= len_target or done:
            break
    env.stop_simulation()
    
    advantages = ac.adv_cal(obs,obs_,dones,rewards,
                            args.GAMMA,args.LAMBDA)
    
    return obs,obs_,acts,dones,rewards,advantages
    
def agent(env,ac,args):
    memo = Replay()
    
    if args.NUM_WORKERS == 1:
        len_target = args.STEPS_PER_EP
    else:
        len_target = args.STEPS_PER_EP / args.NUM_WORKERS # divide tasks to every agent
    
    if args.ONE_TRAJ_FLAG:
        memo.append(*one_trajectory(env,ac,-3e10,len_target,args))
        return memo.return_all()
    while memo.len < len_target:
        memo.append(*one_trajectory(env,ac,memo.len,len_target,args))    
    return memo.return_all()

def train(memos,ac,ac_old,optimizer,args):
    kl_dict = {'max'  :[],
               'mean' :[]}
    
    loss_dict = {'clip'   :[],
                 'entropy' :[],
                 'vf'  :[],
                 'sum' :[]}
    
    for _ in range(args.TRIAN_PER_EP):
        obs,obs_,acts,dones,rewards,advantages = memos.sample()        
        
        # loss 1: loss_clip
        dist_old,log_prob_old, _ = ac_old.get_policy(obs,acts)
        dist_new,log_prob_new, entropies_new = ac.get_policy(obs,acts)
        kl = torch.distributions.kl.kl_divergence(dist_new, dist_old)
        
       
        
        rs = torch.exp(log_prob_new-log_prob_old)
        
        advantages = torch.Tensor(advantages).to(ac.device)
        loss_clip = - torch.minimum(rs*advantages,
                        rs.clip(1-args.EPSILON,1+args.EPSILON)*advantages).mean()
        
        # loss 3: entropy
        loss_entropy = - args.COEF_ENTROPY * entropies_new.mean()
        
        # loss 2: loss_vf
        td_errors = ac.td_cal(obs,obs_,dones,rewards,args.GAMMA)
        loss_vf = args.COEF_VALUE * (td_errors**2).mean()
        
        optimizer.zero_grad()
        (loss_clip+loss_vf+loss_entropy).backward()
        optimizer.step()
        
        kl_dict["max"].append(kl.max().item())
        kl_dict["mean"].append(kl.mean().item())
        loss_dict['clip'].append(loss_clip.item())
        loss_dict['entropy'].append(loss_entropy.item())
        loss_dict['vf'].append(loss_vf.item())
        loss_dict['sum'].append((loss_clip+loss_vf+loss_entropy).item())
        
    return kl_dict,loss_dict

def test(ep,env,ac,args):
    
    env.reset_sim()
    ob = env.get_states()
    
    reward_sum = 0
    step = 0
    while 1:            
        act = ac.get_act(ob)
        
        # env.step()
        act = [400,500,0]
        
        env.action_update(act)
        print(f"Step:{step} \tAct:{act}, \tdistance:{env.reward_distance.value}")
        reward, done = env.get_rewards()     
           
        ob_ = env.get_states()
        
        ob = ob_
        
        step += 1
        reward_sum += reward
        
        if done :
            break
    env.stop_simulation()
    return reward_sum