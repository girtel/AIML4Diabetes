import gym
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario, RandomBalancedScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime
from numba import jit, cuda
from random import seed
from random import randint

env_dict = gym.envs.registration.registry.env_specs.copy()


for env in env_dict:
    if 'simglucose' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

# specify start_time as the beginning of today
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

# --------- Create Random Scenario --------------
# Specify results saving path
path = 'results/results'
seed(99)


# Create a simulation environment
patients = ['child#001','child#002','child#003','child#004','child#005','child#006','child#007','child#008','child#009','child#010',
            'adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010',
            'adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007',
            'adolescent#008','adolescent#009','adolescent#010']
            
def bbsim():
    for patient in patients:
        patient = T1DPatient.withName(patient)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        for seed in range (10,20):
            scenario = RandomScenario(start_time=start_time, seed=randint(10, 99999))
            env = T1DSimEnv(patient, sensor, pump, scenario)
            # Create a controller
            controller = BBController()
            
            # Put them together to create a simulation object
            s1 = SimObj(env, controller, timedelta(days=10), animate=False, path=path+str(seed))
            results1 = sim(s1)
            print('Complete:',patient.name,'-',seed)
    print('All done!')
if __name__=="__main__":
    bbsim()