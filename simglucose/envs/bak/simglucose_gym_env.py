from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.simulation.env import T1DSimEnvExtendedObs as _T1DSimEnvExtendedObs
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        print('Local sg test env')
        if patient_name is None:
            patient_name = 'adolescent#001'
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        s=self.seed()
        print('seeds', s)

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        else:
            return self.env.step(act, reward_fun=self.reward_fun)

    def reset(self):
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        print('_seed called')
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2021, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        self.env = _T1DSimEnv(patient, sensor, pump, scenario)
        return [seed1, seed2, seed3]

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))
class T1DSimEnvExtendedObs(T1DSimEnv):
    def __init__(self, patient_name=None, reward_fun=None, n_samples=20):
        print('Local sg extended observation test env')
        self._n_samples=n_samples
        super(T1DSimEnvExtendedObs,self).__init__(patient_name,reward_fun)
    def seed(self, seed=None):
        print('extended obs _seed called')
        self.np_random, seed1 = seeding.np_random(seed=seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2021, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        self.env = _T1DSimEnvExtendedObs(patient, sensor, pump, scenario,self._n_samples)
        return [seed1, seed2, seed3]
    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,self._n_samples,))

