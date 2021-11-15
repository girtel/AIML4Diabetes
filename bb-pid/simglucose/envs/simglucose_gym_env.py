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

    def __init__(self, patient_name=None, reward_fun=None, seed=None, normalize=False, discrete=False, max_basal=None, state=None, noise=True,  sequence=-1, append_time=False, harrison_benedict=True):
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
        print('Local simglucose test env')
        if patient_name is None:
            patient_name = 'adolescent#001'
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.noise=noise
        self.max_basal=max_basal
        self.sequence=sequence
        self.harrison_benedict=harrison_benedict
        self.np_random, myseed = seeding.np_random(seed=seed)
        print(myseed)
        self.saved_state=None
        self.env, _, _,_,_ = self._create_env_from_random_state()
        self._normalize = normalize
        self._discrete=discrete
        self.append_time=append_time
        #print(self._normalize)
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
          #print('A=',_action,'O=',obs.CGM,'s=',self.episode_steps)
        act = Action(basal=_action, bolus=0)
        if self.reward_fun is None:
          _obs, reward, done , info =  self.env.step(act)
        else:
          _obs, reward, done , info = self.env.step(act, reward_fun=self.reward_fun)
        #print('Au=',action,'A=',_action,'O=',obs.CGM,'r=',reward,'s=',self.episode_steps)
        #_obs, r, done, info = self.env.step(act)
        if not done and self.sequence>0:
           for i in range(1, self.sequence-1):
               #Remember: we pass unnormalized actions to the simglucose env
               act=Action(basal=0,bolus=0)
               #_obs, r, d, info = self.env.step(act)
               _obs, r, d, info = self.env.step(act,reward_fun=self.reward_fun)
               #Make sure that one done=True is not replaced by a later done=False
               reward += r
               if d==True:
                   done=d
                   break
        #if done:
        #   reward=-100 
        #else:
        #    _obs, reward, done, info = self.env.step(act,reward_fun=self.reward_fun)
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

        #print('An=',action,'On=',obs,'A=',_action,'O=',_obs,'r=',reward,'s=',self.episode_steps)
        #print('time=',self.env.time, 'hour', (self.env.time.hour*60 + self.env.time.minute))
        if self.save_callback is not None:
           go_on=self.save_callback.step(self.episode_steps)
           if not go_on:
              info['Time-limit truncated at callback'] = not done
              print('Time-limit truncated from callback at step', self.episode_steps)
              done=True
        self.episode_steps +=1
        #return ([obs], reward, done, info)
        return (np.array([obs]), reward, done, info)


    def tomin(self):
       return (self.env.time.hour*60 + self.env.time.minute)
    def reset(self):

        self.env, seed2,seed3,seed4, hour = self._create_env_from_random_state()
        if self.save_callback is not None:
           self.save_callback.reset([seed2,seed3,seed4,hour])
        obs, _, _, _ = self.env.reset()
        #print('reset(). episode_steps:',self.episode_steps,'obs=',obs)
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

        print('Action space=',self.action_space)
        print('Obs space=',self.observation_space)
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
   # @property
   # def action_space(self):
   #     ub = self.env.pump._params['max_basal']
   #     return spaces.Box(low=0, high=ub, shape=(1,))

   # @property
   # def observation_space(self):
   #     return spaces.Box(low=0, high=np.inf, shape=(1,))

class T1DSimEnvSequentialObs(T1DSimEnv):
    def __init__(self, patient_name=None, reward_fun=None, n_samples=10, normalize=False, max_basal=None, seed=None, noise=True, discrete=False):
        '''
        Provides last n_samples of CGM as observation instead of just the last one, but they do not overlap
        '''
        print('Local simglucose sequential observation test gym env')
        self._n_samples=n_samples
        self.discrete=discrete
        self.obs_vector=np.zeros(n_samples)
        print(self.obs_vector)
        self.n_samples=n_samples 
        super(T1DSimEnvSequentialObs,self).__init__(patient_name=patient_name,reward_fun=reward_fun,normalize= normalize,max_basal=max_basal, seed=seed, noise=noise, discrete=discrete)
    def reset(self):
        obs=super().reset()
        print(obs)
        print(obs[0])
        print(self.obs_vector)
        print(self.obs_vector[0])
        self.obs_vector[0]=obs[0]
        reward=0
        done=False
        print('reset(). episode_steps:',self.episode_steps,'obs=',self.sequence)
        self.episode_steps =0
        #Warning: assuming that done cannot be true again during samples
        for i in range(1,self.n_samples):
            obs, r, done, info = super().step(self.action_space.low)
            if done:
                raise Exception('T1DSimEnvSequentialObs: done during reset. See the number of samples or change the implementation')
            self.obs_vector[i]=obs
            reward += r
            
        print('reset(). obs=',self.sequence)
        self.episode_steps =0
        return self.obs_vector
    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.obs_vector[0]=obs
        for i in range(1, self.n_samples):
            obs, r, d, info = super().step(self.action_space.low)
            #Make sure that one done=True is not replaced by a later done=False
            if not done:
                done=d
            self.obs_vector[i]=obs
            reward += r
        #print('A=',action,'O=',self.sequence,'r=',reward)
        return (self.obs_vector, reward, done, info)
    def _set_spaces(self):
        if self.discrete:
                raise Exception('T1DSimEnvSequentialObs: set_spaces: discrete is not implemented yet')
        
        self.min_action=0
        if self.max_basal is None:
            self.max_action= self.env.pump._params['max_basal']
        else:
            self.max_action= self.max_basal 
        self.low_state = np.repeat(0,self._n_samples)
        self.high_state = np.repeat(np.inf,self._n_samples)
        self.min_obs=0
        #Limit to apply normalization
        self.max_obs=500
        print ('Called set spaces sequential observation')
        if self._normalize:
          print ('normalize')
          self.action_space = spaces.Box(
              low=-1.0,
              high=1.0,
              shape=(1,),
              dtype=np.float32
          )
          self.observation_space = spaces.Box(
              low=np.repeat(-1.0,self._n_samples).astype('float32'),
              high=np.repeat(1.0,self._n_samples).astype('float32'),
              dtype=np.float32
          )

        else:
          self.action_space = spaces.Box(
              low=self.min_action,
              high=self.max_action,
              shape=(1,),
              dtype=np.float32
          )
          self.observation_space = spaces.Box(
              low=self.low_state,
              high=self.high_state,
              dtype=np.float32
          )
        print(self.action_space)
        print(self.observation_space)

class T1DSimEnvExtendedObs(T1DSimEnv):
    def __init__(self, patient_name=None, reward_fun=None, n_samples=20, normalize=False, max_basal=None, noise=True, seed=None, limit_time=-1, append_action=False, append_time=False):
        '''
        Provides last n_samples of CGM as observation instead of just the last one
        '''
        print('Local simglucose extended observation test gym env')
        self._n_samples=n_samples
        self.noise=noise
        self.limit_time=limit_time
        self.append_action=append_action
        
        if limit_time>0:
           print("Time limited to (hours)", (limit_time/60)) 
	
        super(T1DSimEnvExtendedObs,self).__init__(patient_name=patient_name,reward_fun=reward_fun,normalize= normalize,max_basal=max_basal, noise=noise,seed=seed, append_time=append_time)

        print('My obs space is', self.observation_space)
   # @property
   # def observation_space(self):
   #     return spaces.Box(low=0, high=np.inf, shape=(self._n_samples,))
    def _create_env_from_random_state(self):
        #print('_create_env_from_random_state simglucose extended observation')
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
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnvExtendedObs(patient, sensor, pump, scenario,n_samples=self._n_samples, append_action=self.append_action )
        self.time_per_step=int(sensor.sample_time)
        self.time_in_env=0
        return env, seed2, seed3, seed4, hour
    def _set_spaces(self):
        self.min_action=0
        if self.max_basal is None:
            self.max_action= self.env.pump._params['max_basal']
        else:
            self.max_action= self.max_basal 
        ss=self._n_samples
        if self.append_action:
           ss=2*ss
        if self.append_time:
           ss=ss+1
        self.low_state = np.repeat(0,ss)
        self.high_state = np.repeat(np.inf,ss)
        self.min_obs=0
        #Limit to apply normalization
        self.max_obs=500
        print ('Called set spaces extended')
        if self._normalize:
          print ('normalize')
          self.action_space = spaces.Box(
              low=-1.0,
              high=1.0,
              shape=(1,),
              dtype=np.float32
          )
          self.observation_space = spaces.Box(
              low=np.repeat(-1.0,ss).astype('float32'),
              high=np.repeat(1.0,ss).astype('float32'),
              dtype=np.float32
          )

        else:
          self.action_space = spaces.Box(
              low=self.min_action,
              high=self.max_action,
              shape=(1,),
              dtype=np.float32
          )
          self.observation_space = spaces.Box(
              low=self.low_state,
              high=self.high_state,
              dtype=np.float32
          )
        print(self.action_space)
        print(self.observation_space)
    def step(self, action):
        if self._normalize:
          _action=self.rescale_action(action)
        else:
          _action=action
          #print('A=',_action,'O=',obs.CGM,'s=',self.episode_steps)
        act = Action(basal=_action, bolus=0)
        if self.reward_fun is None:
          _obs, reward, done , info =  self.env.step(act)
        else:
          _obs, reward, done , info = self.env.step(act, reward_fun=self.reward_fun)
        #print('Au=',action,'A=',_action,'O=',obs.CGM,'r=',reward,'s=',self.episode_steps)
        if self.append_time:
           if self._normalize:
              obs=self.normalize_obs(np.append(_obs,self.tomin()))
           else:
              obs=np.append(_obs,self.tomin())
        else:
           if self._normalize:
              obs=self.normalize_obs(_obs)
           else:
              obs=_obs

        #print('An=',action,'On=',obs,'A=',_action,'O=',_obs,'r=',reward,'s=',self.episode_steps)
        #print('time=',self.env.time, 'hour', (self.env.time.hour*60 + self.env.time.minute))
        if self.save_callback is not None:
           go_on=self.save_callback.step(self.episode_steps)
           if not go_on:
              info['Time-limit truncated at callback'] = not done
              print('Time-limit truncated from callback at step', self.episode_steps)
              done=True
        self.episode_steps +=1
        if (self.limit_time>0):
          self.time_in_env +=self.time_per_step
          if (self.time_in_env>self.limit_time) :
                done =True
          else:
                done=False
        return (obs, reward, done, info)

    def reset(self):
        print('episode_steps:',self.episode_steps)
        self.episode_steps =0
        self.env, seed2,seed3,seed4, hour = self._create_env_from_random_state()
        if self.save_callback is not None:
           self.save_callback.reset([seed2,seed3,seed4,hour])
        obs, _, _, _ = self.env.reset()
        if self.append_time:
           obs=np.append(obs,self.tomin())
        if self._normalize:
           obs=self.normalize_obs(obs)
        #print(obs)
        return obs
    def normalize_obs(self,obs):
      if self.append_action:
          ac=self.normalize_action(obs[self._n_samples:2*self._n_samples:1])
          #print('aco',obs[self._n_samples:2*self._n_samples:1],'l',len(obs[-self._n_samples:]),'obs',obs, 'on', obs[:self._n_samples])
          on=((2*obs[:self._n_samples] - self.max_obs -self.min_obs)/(self.max_obs-self.min_obs))
          if self.append_time:
              t=(2*obs[-1:] - 1439)/1439
              #print('on',on,'l',len(on))
              return np.concatenate((on,ac,t),axis=None)
          else:
              return np.concatenate((on,ac),axis=None)

      else:
        if self.append_time:
            on=((2*obs[:self._n_samples] - self.max_obs -self.min_obs)/(self.max_obs-self.min_obs))
            t=(2*obs[-1] - 1439)/1439
            return np.concatenate((on,t),axis=None)
        else:
            return ((2*obs - self.max_obs -self.min_obs)/(self.max_obs-self.min_obs))
class T1DSimGoalEnv(gym.GoalEnv):
    def __init__(self, patient_name=None, reward_fun=None, n_samples=60, normalize=False,max_basal=None, max_steps=16384):
        '''
        Goal-based scenario of simglucose
        '''
        print('Local simglucose Goal extended observation test gym env')
        self._n_samples=n_samples
        self.ext_env=T1DSimEnvExtendedObs(patient_name=patient_name,reward_fun=reward_fun,normalize= normalize, max_basal=max_basal,n_samples=n_samples)
        #self.ext_env=T1DSimEnvSequentialObs(patient_name=patient_name,reward_fun=reward_fun,normalize= normalize, max_basal=max_basal,n_samples=n_samples)
        self.u_goal=np.repeat(112.517,n_samples)
        self.goal=np.repeat(-0.54972,n_samples)
        self.max_episode_steps=max_steps
        self.elapsed_steps=0
        print(self.goal)
        self._set_spaces()
        print('My obs space is', self.observation_space)
       
   # @property
   # def observation_space(self):
   #     return spaces.Box(low=0, high=np.inf, shape=(self._n_samples,))

    def compute_reward(
        self, achieved_goal:  np.ndarray, desired_goal:  np.ndarray, _info: dict
    ) -> np.float32:
        #reward =np.linalg.norm(achieved_goal-desired_goal)
        _, _, risk_current = risk_index(self.ext_env.rescale_obs(achieved_goal), len(achieved_goal))
        #risk_goal = 
        #reward = np.abs(risk_current-risk_goal)
        #print('current', risk_current)
        #print('goal', risk_goal)
        reward=risk_current
        if reward >0.5:
           reward=0
        #print('goal reward=',reward)
        return reward 
    def step(self, action):
        # This gym only controls basal insulin
        obs, reward, done , info =  self.ext_env.step(action)
        self.elapsed_steps +=1
        if self.elapsed_steps >= self.max_episode_steps:
            info['TimeLimit.truncated'] = not done
            print('TimeLimit truncated')
            done = True
          
        gobs = OrderedDict(
            [
                ("observation", obs ),
                ("achieved_goal", obs ),
                ("desired_goal", self.goal),
            ]
        )
        reward=self.compute_reward(obs,self.goal,info)
        
        return (gobs, reward, done, info)

    def reset(self):
        obs = self.ext_env.reset()
        print(obs)
        self.elapsed_steps=0
        gobs = OrderedDict(
            [
                ("observation", obs ),
                ("achieved_goal", obs ),
                ("desired_goal", self.goal),
            ]
        )
        return gobs

    def _set_spaces(self):
        
        self.observation_space = spaces.Dict({
            "observation": self.ext_env.observation_space,
            "achieved_goal":  spaces.Box(
              low=np.repeat(-1.0,self._n_samples).astype('float32'),
              high=np.repeat(1.0,self._n_samples).astype('float32'),
              dtype=np.float32
            ),
            "desired_goal": spaces.Box(
              low=np.repeat(-1.0,self._n_samples).astype('float32'),
              high=np.repeat(1.0,self._n_samples).astype('float32'),
              dtype=np.float32
            )
        })
        self.action_space = spaces.Box(
              low=-1.0,
              high=1.0,
              shape=(1,),
              dtype=np.float32
        )
 
 
        print(self.action_space)
        print(self.observation_space)
    def set_callback(self,callback):
       self.ext_env.set_callback(callback)
    @property
    def time(self):
        return self.ext_env.time
    @property
    def scenario(self):
        return self.ext_env.scenario
    def render(self, mode='human', close=False):
        self.ext_env.render(close=close)

class T1DGroundTruthSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, normalize=False, discrete=False, max_basal=None, state=None, noise=True,  sequence=-1):
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
        print('Local simglucose T1DGroundTruthSimEnv env')
        if patient_name is None:
            patient_name = 'adolescent#001'
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.noise=noise
        self.max_basal=max_basal
        self.sequence=sequence
        self.np_random, myseed = seeding.np_random(seed=seed)
        print(myseed)
        self.saved_state=None
        self.env, _, _,_,_ = self._create_env_from_random_state()
        self._normalize = normalize
        self._discrete=discrete
        print(self._normalize)
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
          #print('A=',_action,'O=',obs.CGM,'s=',self.episode_steps)
        act = Action(basal=_action, bolus=0)
        if self.reward_fun is None:
          _obs, reward, done , info =  self.env.step(act)
        else:
          _obs, reward, done , info = self.env.step(act, reward_fun=self.reward_fun)
        #print('Au=',action,'A=',_action,'O=',obs.CGM,'r=',reward,'s=',self.episode_steps)
        obs=self.env.patient.state
        #if done:
        #   reward=-100 
        #else:
        #    _obs, reward, done, info = self.env.step(act,reward_fun=self.reward_fun)
       

        print('An=',action,'On=',obs,'A=',_action,'O=',_obs,'r=',reward,'s=',self.episode_steps)
        if self.save_callback is not None:
           go_on=self.save_callback.step(self.episode_steps)
           if not go_on:
              info['Time-limit truncated at callback'] = not done
              print('Time-limit truncated from callback at step', self.episode_steps)
              done=True
        self.episode_steps +=1
        return (np.array([obs]), reward, done, info)

    def reset(self):

        self.env, seed2,seed3,seed4, hour = self._create_env_from_random_state()
        if self.save_callback is not None:
           self.save_callback.reset([seed2,seed3,seed4,hour])
        obs, _, _, _ = self.env.reset()
        obs=self.env.patient.state
        print('reset(). episode_steps:',self.episode_steps,'obs=',obs)
        obs=np.array([obs])
        self.episode_steps =0
        return obs

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
        else:
           if self._discrete:
              self.action_space=spaces.Discrete(300)
           else:
              self.action_space=spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        #Not sure how to normalize the obs space
        self.observation_space= spaces.Box(low=0, high=np.inf, shape=(len(self.env.patient.state),))
        print('Action space=',self.action_space)
        print('Obs space=',self.observation_space)
    def normalize_action(self,action):
        return ((2*action - self.max_action -self.min_action)/(self.max_action-self.min_action))

    def rescale_action(self,action):
        return (0.5*(action*(self.max_action-self.min_action) + self.max_action +self.min_action))

    def normalize_obs(self,obs):
        return ((2*obs - self.max_obs -self.min_obs)/(self.max_obs-self.min_obs))

    def rescale_obs(self,obs):
        return (0.5*(obs*(self.max_obs-self.min_obs) + self.max_obs +self.min_obs))
   # @property
   # def action_space(self):
   #     ub = self.env.pump._params['max_basal']
   #     return spaces.Box(low=0, high=ub, shape=(1,))

   # @property
   # def observation_space(self):
   #     return spaces.Box(low=0, high=np.inf, shape=(1,))

