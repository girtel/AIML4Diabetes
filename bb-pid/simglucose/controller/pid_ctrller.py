from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P=1, I=0, D=0, target=140):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = 0

    def policy(self, observation, reward, done, **kwargs):
        #sample_time = kwargs.get('sample_time')
        sample_time =1 
        # BG is the only state for this PID controller
        #bg = observation.CGM
        bg = observation
        print('error', (bg-self.target))
        control_input = self.P * (bg - self.target) + \
            self.I * self.integrated_state + \
            self.D * (bg - self.prev_state) / sample_time

        logger.info('Control input: {}'.format(control_input))

        # update the states
        self.prev_state = bg
        self.integrated_state += (bg - self.target) * sample_time
        logger.info('prev state: {}'.format(self.prev_state))
        logger.info('integrated state: {}'.format(self.integrated_state))

        # return the action
        #action = Action(basal=control_input, bolus=0)
        action = control_input
        print("A=",action)
        return action

    def reset(self):
        self.integrated_state = 0
        self.prev_state = 0


class FoxPIDController(Controller):
    def __init__(self, setpoint, kp, ki, kd, basal=None):

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.basal = basal
        self.setpoint = setpoint
    def policy(self, observation, reward, done, **kwargs):
        value=observation.CGM
        error = self.setpoint - value
        p_act = self.kp * error
        # print('p: {}'.format(p_act))
        self.integral += error
        i_act = self.ki * self.integral
        # print('i: {}'.format(i_act))
        d_act = self.kd * (error - self.previous_error)
        try:
            if self.basal is not None:
                b_act = self.basal
            else:
                b_act = 0
        except:
            b_act = 0
        # print('d: {}'.format(d_act))
        self.previous_error = error
        control_input = p_act + i_act + d_act + b_act
        action = Action(basal=control_input, bolus=0)
        #print('error=', error,'A=',action)
        return action

    def reset(self):
        self.integral = 0
        self.previous_error = 0
