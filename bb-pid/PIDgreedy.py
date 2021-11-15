import gym
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.pid_ctrller import PIDController, FoxPIDController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
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
            'adolescent#001','adolescent#002','adolescent#003','adolescent#004','adolescent#005','adolescent#006','adolescent#007',
            'adolescent#008','adolescent#009','adolescent#010',
            'adult#001','adult#002','adult#003','adult#004','adult#005','adult#006','adult#007','adult#008','adult#009','adult#010',]

pidparams = [[-3.49E-05,-1.00E-07,-1.00E-03],
            [-3.98E-05,-2.87E-08,-3.98E-03],
            [-6.31E-05,-1.74E-08,-1.00E-03],
            [-6.31E-05,-1.00E-07,-1.00E-03],
            [-1.00E-04,-2.87E-08,-6.31E-03],
            [-3.49E-05,-1.00E-07,-1.00E-03],
            [-3.98E-05,-6.07E-08,-2.51E-03],
            [-3.49E-05,-3.68E-08,-1.00E-03],
            [-3.49E-05,-1.00E-07,-1.00E-03],
            [-4.54E-06,-3.68E-08,-2.51E-03],
            [-1.74E-04,-1.00E-07,-1.00E-02],
            [-1.00E-04,-1.00E-07,-6.31E-03],
            [-1.00E-04,-1.00E-07,-3.98E-03],
            [-1.00E-04,-1.00E-07,-4.79E-03],
            [-6.31E-05,-1.00E-07,-6.31E-03],
            [-4.54E-10,-1.58E-11,-1.00E-02],
            [-1.07E-07,-6.07E-08,-6.31E-03],
            [-4.54E-10,-4.54E-12,-1.00E-02],
            [-6.31E-05,-1.00E-07,-3.98E-03],
            [-4.54E-10,-4.54E-12,-1.00E-02],
            [-1.58E-04,-1.00E-07,-1.00E-02],
            [-3.98E-04,-1.00E-07,-1.00E-02],
            [-4.54E-10,-1.00E-07,-1.00E-02],
            [-1.00E-04,-1.00E-07,-3.98E-03],
            [-3.02E-04,-1.00E-07,-1.00E-02],
            [-2.51E-04,-2.51E-07,-1.00E-02],
            [-1.22E-04,-3.49E-07,-2.87E-03],
            [-1.00E-04,-1.00E-07,-1.00E-02],
            [-1.00E-04,-1.00E-07,-1.00E-02],
            [-1.00E-04,-1.00E-07,-1.00E-02]]

def pidsim():
    for idx,patient in enumerate(patients):
        patient = T1DPatient.withName(patient)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        p,i,d = pidparams[idx]
        for seed in range (10,20):
            scenario = RandomScenario(start_time=start_time, seed=randint(10, 99999))
            env = T1DSimEnv(patient, sensor, pump, scenario)
            # Create a controller
            controller = FoxPIDController(112.517,kp=p, ki=i, kd=d)
            # Put them together to create a simulation object
            s1 = SimObj(env, controller, timedelta(days=10), animate=False, path=path+str(seed))
            results1 = sim(s1)
            print('Complete:',patient.name,'-',seed)
    print('All done!')
if __name__=="__main__":
    pidsim()