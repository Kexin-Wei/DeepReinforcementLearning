import vrep
import time
import numpy as np
import vrep_user_lib as vu
import vrep as v

class Lapra_Sim_dynamic_single:
    def __init__(self, timestep, port, fast=False,time_episode = 0.1):
        
        # intializing variables
        self.timestep = timestep
        self.time_episode = time_episode
        self.total_time = 0
        self.done = 0
        self.reward = 0
        self.port = port
        self.fast = fast
        
        #Initializing V-Rep connection and obtaining simulation time step
        _, self.clientID = vu.initialize_Vrep_connection(True, self.port)
        
        # intializing device and dummmy structures
        self.device, self.tip, self.target = vu.initialize_single_device(self.clientID)
        # initializing distance structure
        self.reward_distance = vu.Distance('distance_to_target',None,None)
        
        # initializing position and force streams
        vu.initialize_dummy_position_stream(self.clientID,self.tip)
        vu.initialize_dummy_position_stream(self.clientID,self.target)
        vu.initialize_distance_stream(self.clientID,self.reward_distance)
        vu.initialize_device_streams(self.clientID,self.device)
        
        # initializing agent input states 
        self.input = vu.User_input()
        
        # Associate joint handles and forces with user inputs
        # Need to matches the handles from self.device.joints
        #self.input.associate_handles([43,34,94,103,116])
##        self.input.associate_handles([42,33,93,102,115])
        self.active_handles = [20,33,64,93,115]
        self.name_handles = ['yaw','pitch','insertion','roll','gripper']
        self.input.associate_handles(self.active_handles)
        self.forces = [5,5,3,5,1e-4]
        self.input.associate_forces([ x*40 for x in self.forces])
        # Random target
        self.move_target_random()



    def action_update(self, action):

        self.current_action = action
##        action = np.clip(action, -2, 2)
##        self.current_action = action.tolist()
        
        # apply forces according to actions
##        for joint in self.device.joints:
##            DOFs_passed = 0
##            for i,DOF in enumerate(self.input.inputs[-1]):
##                if DOFs_passed < 4: 
##                    if joint.handle == self.input.inputs[DOF].handle:
##                        applied_force = self.input.inputs[DOF].force * action[i]
##                        vu.apply_force(self.clientID,applied_force,joint.handle)
##                        DOFs_passed += 1
##                    else:
##                        break
        for joint in self.device.joints:
            if joint.handle in self.active_handles[:-2]:
                idx = self.active_handles.index(joint.handle)
                applied_force = self.input.inputs[self.name_handles[idx]].force \
                                * action[idx]
##                vu.apply_force(self.clientID,applied_force,joint.handle)
                v.simxSetJointForce(self.clientID,\
                                    joint.handle,\
                                    200,\
                                    v.simx_opmode_oneshot)
                v.simxSetJointTargetVelocity(self.clientID,joint.handle,\
                                             applied_force,\
                                             v.simx_opmode_oneshot)
                
                        
        self.total_time += self.timestep
        
        # execute simulation step in V-Rep and wait for it to finish
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        # get positions, forces
        vu.update_device(self.clientID,self.device)
        vu.update_dummy_position(self.clientID,self.target)
        vu.update_dummy_position(self.clientID,self.tip)
        vu.update_distance(self.clientID,self.reward_distance)
        


    def get_rewards(self):

        if self.reward_distance.value <= 0.015:
            self.reward += 200
            self.done = 1
            #print('Reached target')
            return self.reward - self.total_time, self.done
        
        if (self.old_dist > self.reward_distance.value):
            self.reward = 0.05 + \
                          0.1 * np.exp(-self.reward_distance.value)
        else:
            self.reward = -0.05
##                            0.1 * np.exp(self.reward_distance.value))
#        d = self.reward_distance.value
#        self.reward = 1/np.sqrt(d)+np.exp(-d)

        if self.total_time >= self.time_episode:
            self.done = 1

        self.old_dist = self.reward_distance.value

        return self.reward, self.done



    def get_states(self):
##        states = np.array([])
##        for joint in self.device.joints:
##            if joint.handle in self.active_handles[:-1]:
##                state = joint.force
##                states = np.append(states,state)
##
##        for obj in [self.tip,self.target]:
##            states = np.append(states,obj.position)
        states = []
        states.extend(self.current_action)
        for obj in [self.tip,self.target]:
            states.extend(obj.position)
        return np.array(states)[np.newaxis,:]

        
     
    def reset_sim(self):
        # set joint forces to 0
        for joint in self.device.joints:
            joint.force = 0.
        
        # stop simulation
##        _ = vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        self.stop_simulation()
        time.sleep(1)
        vrep.simxSynchronous(self.clientID, True)
        

##        # reset forces to 0 (again?)
##        for joint in self.device.joints:
##            DOFs_passed = 0
##            for i,DOF in enumerate(self.input.inputs):
##                DOFs_passed += 1
##                if DOFs_passed < 5: 
##                    if joint.handle == self.input.inputs[DOF].handle:
##                        vu.apply_force(self.clientID,0,joint.handle)
                        
        # re-initialize position, force streams
        vu.initialize_dummy_position_stream(self.clientID,self.tip)
        vu.initialize_dummy_position_stream(self.clientID,self.target)
        vu.initialize_distance_stream(self.clientID,self.reward_distance)
        vu.initialize_device_streams(self.clientID,self.device)

        # execute step in new simulation
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)

        # Start simulation
##        _ =vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        self.move_target_random()
        self.start_simulation()

        self.total_time = 0
        self.done = 0
        self.reward = 0
        
        
    def move_target_random(self):
#         r = np.random.uniform(low=0.05,high=0.1,size=None)
# ##        r = np.random.uniform(low=0.05,high=0.181575,size=None)
#         theta = np.random.uniform(low = 0, high = np.pi*2,size=None)
#         # random vector + tip coordinate
#         x = r*np.cos(theta) + (-0.0435)
#         y = r*np.sin(theta) + (-0.00053)
#         z = np.random.uniform(low=1.280e-2,high=9.65e-2,size=None)
        x = -0.1
        y = -0.1
        z = 0
        pos = [x,y,z]
        # print(f"Target position {pos}")
        v.simxSetObjectPosition(self.clientID,self.target.handle,-1,pos,\
                                v.simx_opmode_oneshot)


    def start_simulation(self):
        vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)
        if self.fast:
            vrep.simxSetBooleanParameter(self.clientID,
                                         vrep.sim_boolparam_display_enabled,
                                         False,
                                         vrep.simx_opmode_oneshot)
        #take dummy action to get distance right
        self.action_update([0,0,0])
        self.old_dist = self.reward_distance.value
        
    def stop_simulation(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
        
    def close_simulation(self):
        vrep.simxFinish(self.clientID)
