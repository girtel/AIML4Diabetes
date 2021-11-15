import numpy as np
from tf_agents.metrics import py_metric
from  tf_agents.utils.numpy_storage import NumpyState
from simglucose.analysis.risk import risk_index
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.utils import common


'''This is a PyMetric. It will not work with tf metrics or anything computed in graph mode...I think'''
#TODO: try to implement as TF metric or wrap as TFPyMetric... 
#TODO: do this without appending 
class AverageRiskPyMetric(py_metric.PyStepMetric):
  def __init__(self, name='AverageRiskPyMetric', show=False, min_obs=0,max_obs=500):
      super(py_metric.PyStepMetric, self).__init__(name)
      self._np_state = NumpyState()
      self.reset()
      self._show=show
      self.min_obs=min_obs
      self.max_obs=max_obs
  def reset(self):
      self._np_state.risk=np.array([], dtype=np.float32)
      self._np_state.total=np.array([], dtype=np.float32)
      self._np_state.hbgi=np.array([], dtype=np.float32)
      self._np_state.total_hbgi=np.array([], dtype=np.float32)
      self._np_state.lbgi=np.array([], dtype=np.float32)
      self._np_state.total_lbgi=np.array([], dtype=np.float32)
  def call(self, trajectory):
      lbgi, hbgi, ri=risk_index(self.rescale_obs(trajectory.observation[0]),1)
      if self._show:
        print("RI_metric obs",trajectory.observation[0])
        print("RI_metric reward=",trajectory.reward)
        print("RI_metric action=",trajectory.action)
        print("RI_metric info=",trajectory.policy_info)
        print("RI_metric ri=",ri)
        print("RI_metric is_first=",trajectory.is_first())
        print("RI_metric is_boundary=",trajectory.is_boundary())
      if not trajectory.is_boundary():
         self._np_state.risk=np.append(self._np_state.risk,ri)
         self._np_state.lbgi=np.append(self._np_state.lbgi,lbgi)
         self._np_state.hbgi=np.append(self._np_state.hbgi,hbgi)
      if trajectory.is_last():
       self._np_state.total=np.append(self._np_state.total,np.mean(self._np_state.risk))
       self._np_state.total_lbgi=np.append(self._np_state.total_lbgi,np.mean(self._np_state.lbgi))
       self._np_state.total_hbgi=np.append(self._np_state.total_hbgi,np.mean(self._np_state.hbgi))
       self._np_state.risk=np.array([], dtype=np.float32)
       self._np_state.lbgi=np.array([], dtype=np.float32)
       self._np_state.hbgi=np.array([], dtype=np.float32)
      return trajectory
  def result(self):
      print('av lbgi=',np.mean(self._np_state.total_lbgi),'av hbgi=',np.mean(self._np_state.total_hbgi)) 
      return (np.mean(self._np_state.total))
  def rescale_obs(self,obs):
      return (0.5*(obs*(self.max_obs-self.min_obs) + self.max_obs +self.min_obs))

class EpisodeLoggerPyMetric(py_metric.PyStepMetric):
  def __init__(self, name='EpisodeLoggerPyMetric'):
      super(py_metric.PyStepMetric, self).__init__(name)
      self._np_state = NumpyState()
      self.reset()
  def reset(self):
      self._np_state.steps = np.int64(0)
      self._np_state.episode = np.int64(0)
      print("Resetting EpisodeLoggerPyMetric")
  def call(self, trajectory):
      if trajectory.is_first():
       print ("Starting episode ",self._np_state.episode)
       self._np_state.episode +=1
      print("\tStep", self._np_state.steps)
      self._np_state.steps +=1
      _, _, ri=risk_index(trajectory.observation[0],1)
      print("\t\tobs=", trajectory.observation[0])
      print("\t\treward=", trajectory.reward[0])
      print("\t\tri=", ri)
      if trajectory.is_last():
       print ("Ending episode ",self._np_state.episode)
       self._np_state.steps = np.int64(0) 
      return trajectory
  def result(self):
      print("Called result espisode logger. Count=",self._np_state.episode)
      return self._np_state.episode

class RewardHistogram(tf_metric.TFHistogramStepMetric):
  """Metric to compute the frequency of rewards."""

  def __init__(self,
               name='RewardHistogram',
               dtype=tf.int32,
               buffer_size=100):
    super(RewardHistogram, self).__init__(name=name)
    self._buffer = TFDeque(buffer_size, dtype)
    self._dtype = dtype

  @common.function
  def call(self, trajectory):
    self._buffer.extend(trajectory.reward)
    return trajectory

  @common.function
  def result(self):
    return self._buffer.data

  @common.function
  def reset(self):
    self._buffer.clear()

class AverageBGPyMetric(py_metric.PyStepMetric):
  def __init__(self, name='AverageBGPyMetric', show=False):
      super(py_metric.PyStepMetric, self).__init__(name)
      self._np_state = NumpyState()
      self.reset()
      self._show=show
  def reset(self):
      self._np_state.bg=np.array([], dtype=np.float32)
      self._np_state.total=np.array([], dtype=np.float32)
  def call(self, trajectory):
      if self._show:
        print("BG_metric obs",trajectory.observation[0])
        print("BG_metric reward=",trajectory.reward)
        print("BG_metric action=",trajectory.action)
        print("BG_metric is_first=",trajectory.is_first())
        print("BG_metric is_boundary=",trajectory.is_boundary())
      if not trajectory.is_boundary():
         self._np_state.bg=np.append(self._np_state.bg,trajectory.observation[0])
      if trajectory.is_last():
       self._np_state.total=np.append(self._np_state.total,np.mean(self._np_state.bg))
       self._np_state.bg=np.array([], dtype=np.float32)
      return trajectory
  def result(self):
      return (np.mean(self._np_state.total))

class EndBGPyMetric(py_metric.PyStepMetric):
  def __init__(self, name='EndBGPyMetric', show=False):
      super(py_metric.PyStepMetric, self).__init__(name)
      self._np_state = NumpyState()
      self.reset()
      self._show=show
  def reset(self):
      self._np_state.bg=np.float32(0)
  def call(self, trajectory):
      if self._show:
        print("BGEnd_metric obs",trajectory.observation[0])
        print("BGEnd_metric reward=",trajectory.reward)
        print("BGEnd_metric action=",trajectory.action)
        print("BGEnd_metric ri=",ri)
        print("BGEnd_metric is_first=",trajectory.is_first())
        print("BGEnd_metric is_boundary=",trajectory.is_boundary())
      if trajectory.is_last():
       self._np_state.bg=trajectory.observation.numpy()
      return trajectory
  def result(self):
      return self._np_state.bg



