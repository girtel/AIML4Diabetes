from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.simulation.env import T1DSimEnvExtendedObs as _T1DSimEnvExtendedObs
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario_gen import RandomBalancedScenario
from simglucose.controller.base import Action
from simglucose.analysis.risk import risk_index
import numpy as np
import pandas as pd
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
from typing import Dict
from collections import OrderedDict
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, normalize=False, discrete=False, max_basal=None, state=None, noise=True,  sequence=-1, append_time=False, harrison_benedict = False):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        normalize: normalize observations and actions to [-1,1]
        discrete: use discrete insulin dosis. TODO: should pass the ranges, now it is harcoded in set_spaces and step
        state: pass a vector with the seed an hour values to reproduce a particular env
        noise: if False, the real BG instead of the CGM is used in the observations
        sequence: the number of steps to run with zero insulin before a new observation is returned (-1 if every step returns an observation)
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = 'adolescent#001'
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.noise=noise
        self.max_basal=max_basal
        self.sequence=sequence
        self.harrison_benedict=harrison_benedict
        self.np_random, myseed = seeding.np_random(seed=seed)
        self.saved_state=None
        self.env, _, _,_,_ = self._create_env_from_random_state()
        self._normalize = normalize
        self._discrete=discrete
        self.append_time=append_time
	#Set the spaces
        self._set_spaces()
        self.episode_steps=0
        self.save_callback = None
        self.saved_state=state
    def set_callback(self,callback):
        self.save_callback=callback
    def step(self, action):
        # This gym only controls basal insulin
        if self._normalize:
          _action=self.rescale_action(action)
          if self._discrete:
            _action=action/5
        else:
          _action=action
        act = Action(basal=_action, bolus=0)
        if self.reward_fun is None:
          _obs, reward, done , info =  self.env.step(act)
        else:
          _obs, reward, done , info = self.env.step(act, reward_fun=self.reward_fun)
        if not done and self.sequence>0:
           for i in range(1, self.sequence-1):
               #Remember: we pass unnormalized actions to the simglucose env
               act=Action(basal=0,bolus=0)
               _obs, r, d, info = self.env.step(act,reward_fun=self.reward_fun)
               #Make sure that one done=True is not replaced by a later done=False
               reward += r
               if d==True:
                   done=d
                   break
        if self.append_time:
            if self._normalize:
               obs=self.normalize_obs(np.array([_obs.CGM,self.tomin()]))
            else:
               obs=np.array([_obs.CGM,self.tomin()])
        else:
            if self._normalize:
               obs=self.normalize_obs(_obs.CGM)
            else:
               obs=_obs.CGM

        if self.save_callback is not None:
           go_on=self.save_callback.step(self.episode_steps)
           if not go_on:
              info['Time-limit truncated at callback'] = not done
              done=True
        self.episode_steps +=1
        return (np.array([obs]), reward, done, info)


    def tomin(self):
       return (self.env.time.hour*60 + self.env.time.minute)
    def reset(self):

        self.env, seed2,seed3,seed4, hour = self._create_env_from_random_state()
        if self.save_callback is not None:
           self.save_callback.reset([seed2,seed3,seed4,hour])
        obs, _, _, _ = self.env.reset()
        if self.append_time:
            if self._normalize:
               obs=self.normalize_obs(np.array([obs.CGM,self.tomin()]))
            else:
               obs=np.array([obs.CGM,self.tomin()])
        else:
            if self._normalize:
               obs=self.normalize_obs(obs.CGM)
            else:
               obs=obs.CGM
        if self._normalize:
           obs=self.normalize_obs(obs)
        self.episode_steps =0
        return np.array([obs])

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3 = self._create_env_from_random_state()
        return [seed1, seed2, seed3]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31
        hour = self.np_random.randint(low=0.0, high=24.0)
        if self.saved_state is not None:
           seed2=self.saved_state[0]
           seed3=self.saved_state[1]
           seed4=self.saved_state[2]
           hour=self.saved_state[3]
           print('Using state', seed2, seed3, seed4, hour)
        start_time = datetime(2021, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        if self.harrison_benedict:
            vpatient_params = pd.read_csv(PATIENT_PARA_FILE)
            self.kind = self.patient_name.split('#')[0]
            self.bw = vpatient_params.query('Name=="{}"'.format(self.patient_name))['BW'].item()
            scenario=RandomBalancedScenario(start_time=start_time, seed=seed3, harrison_benedict=self.harrison_benedict, bw=self.bw, kind=self.kind)
        else:
            scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario,noise=self.noise)
        return env, seed2, seed3, seed4, hour

    def render(self, mode='human', close=False):
        self.env.render(close=close)
    @property
    def time(self):
        return self.env.time
    @property
    def scenario(self):
        return self.env.scenario
    def _set_spaces(self):
        self.min_action=0
        if self.max_basal is None:
            self.max_action= self.env.pump._params['max_basal']
        else:
            self.max_action= self.max_basal 
        self.min_obs=0
        #Limit to apply normalization
        self.max_obs=500
        if self._normalize:
           if self._discrete:
              self.action_space=spaces.Discrete(150)
           else:
              self.action_space=spaces.Box(low=-1, high=1, shape=(1,))
           if self.append_time:
              self.observation_space= spaces.Box(low=-1, high=np.array([1,1]))
           else:
              self.observation_space= spaces.Box(low=-1, high=1, shape=(1,))
        else:
           if self._discrete:
              self.action_space=spaces.Discrete(300)
           else:
              self.action_space=spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
           if self.append_time:
              self.observation_space= spaces.Box(low=0, high=np.array([np.inf, 1439]))
           else:
              self.observation_space= spaces.Box(low=0, high=np.inf, shape=(1,))

    def normalize_action(self,action):
        return ((2*action - self.max_action -self.min_action)/(self.max_action-self.min_action))

    def rescale_action(self,action):
        return (0.5*(action*(self.max_action-self.min_action) + self.max_action +self.min_action))

    def normalize_obs(self,obs):
        if self.append_time:
           o=(2*obs[0] - self.max_obs -self.min_obs)/(self.max_obs-self.min_obs)
           t=(2*obs[1] - 1439)/1439
           return np.array([o,t])
        else:
           return ((2*obs - self.max_obs -self.min_obs)/(self.max_obs-self.min_obs))

    def rescale_obs(self,obs):
        return (0.5*(obs*(self.max_obs-self.min_obs) + self.max_obs +self.min_obs))