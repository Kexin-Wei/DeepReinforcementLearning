import numpy as np
from math import e
from tensorflow.compat.v1.keras.backend import set_session

import numpy as np
from math import e
from tensorflow.compat.v1.keras.backend import set_session

class imagine_trajectory:

    def __init__(self, session, graph, time_step, time_episode, dynamic_model, policy, steps):
        self.time_episode = time_episode
        self.time_step = time_step
        self.policy = policy
        self.model = dynamic_model
        self.steps = steps
        self.graph = graph
        self.session = session

    def roll_out(self, seed_state, seed_state_idx, steps):
        """
        seed_states: 1-by-m numpy array contains state vector for imaginary rollout
        seed_states_idx: time index of seed states
        """
        obs_list = []
        action_list = []
        rew_list = []
        obs_next_list = []
        vpred_list = []
        obs_info_list = []
        input_state = seed_state
        for step in range(self.steps):
            action, v_pred = self.policy.act(obs=input_state, stochastic=True)
            model_input = np.append(input_state,action)[np.newaxis,:]
            with self.graph.as_default():
                set_session(self.session)
                next_state = self.model.predict(model_input)
            
            time = (seed_state_idx + step + 1) * self.time_step
            reward, done = self.get_rewards(input_state, next_state[0], time)
            
            obs_list.append(input_state)
            action_list.append(action)
            rew_list.append(reward)
            obs_next_list.append(next_state)
            vpred_list.append(v_pred)
            obs_info_list.append(1-done)
            input_state = next_state
            if done:
                break
        if done:
            v_next = 0
        else:
            _, v_next = self.policy.act(obs=input_state, stochastic=True)
        vpred_list.append(v_next)         
        return obs_list, action_list, rew_list, obs_next_list, vpred_list[:-1], vpred_list[1:], obs_info_list


    def get_rewards(self, seed_state, next_state, time):
        done = 0
        reward_distance = (((next_state[-6:-3] - next_state[-3:])**2).sum())**0.5
        old_dist = (((seed_state[-6:-3] - seed_state[-3:])**2).sum())**0.5
        if (old_dist > reward_distance):
            self.reward = 0.05+\
                          0.1 * np.exp(-reward_distance)
        else:
            self.reward = -0.05
##                            0.1 * np.exp(reward_distance))

        if reward_distance <= 0.008:
            self.reward += 200
            return self.reward - time, True

        if time >= self.time_episode:
            done = 1

        return self.reward, done
