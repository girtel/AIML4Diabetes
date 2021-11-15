r"""Eval RNN PPO from a loaded policy.

To run:

```bash
python -u ppo-rnn-load.py --root_dir=DIR_TO_LOG_METRICS --saved_dir=DIR_WHERE_SAVED_POLICY_IS
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging

import gin
from six.moves import range
import tensorflow as tf 

import tensorflow_probability as tfp

from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
#Add it to save our eval policy
from tf_agents.policies import policy_saver

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.wrappers import TimeLimit
from simglucose.envs import T1DSimEnv 
from simglucose.analysis.risk import risk_index
#Our custom metrics
from sg_metric import AverageRiskPyMetric
from sg_metric import AverageBGPyMetric
#Our custom functions. 
from sg_rewards import stepReward3_eval


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for loading policy and saving results.')
flags.DEFINE_string('saved_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for loading policy and saving results.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    saved_dir,
    env_name='simglucose',
    task_name='balance',
    observations_whitelist='position',
    eval_env_name=None,
    num_iterations=1000000,
    num_parallel_environments=1,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=20,
    eval_interval=10000,
    # Params for summaries and logging
    log_interval=100,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):
  """A simple eval for RNN SAC on Diabetes control."""


  root_dir = os.path.expanduser(root_dir)
  saved_model_dir = os.path.expanduser(saved_dir)

  summary_writer = tf.compat.v2.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

  eval_metrics = [
      AverageBGPyMetric(), #Add our custom metric. Since it is not TF metric, we do not have to allow graph execution
      AverageRiskPyMetric(show=True), #Add our custom metric. Since it is not TF metric, we do not have to allow graph execution
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()

  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    logging.info("Creating simglucose environments")
    eval_tf_env = tf_py_environment.TFPyEnvironment(TimeLimit(GymWrapper(T1DSimEnv(patient_name='adult#008',reward_fun=stepReward3_eval,normalize=True,sequence=15,harrison_benedict=True)),350))
    time_step_spec = eval_tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = eval_tf_env.action_spec()


    env_steps = tf_metrics.EnvironmentSteps(prefix='Eval')
    
    #It is necessary to use as_composite, otherwise it cannot load the policy. First we have to create the distribution we are passing to as_composite
    #which is DeterministicWithLogProb
    eval_policy = tf.compat.v2.saved_model.load(saved_dir)
    #With this function we run the environment with the policy for num_eval_episodes
    results = metric_utils.eager_compute(
              eval_metrics,
              eval_tf_env,
              eval_policy,
              num_episodes=num_eval_episodes,
              train_step=env_steps.result(),
              summary_writer=summary_writer,
              summary_prefix='Eval',
              use_function=False #Do not use function, otherwise it uses graph mode and our py metrics does not work. To improve performance this should be changed
    )
    metric_utils.log_metrics(eval_metrics)

    global_step_val = global_step.numpy()



def main(_):
  tf.compat.v1.enable_v2_behavior()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
   try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(FLAGS.root_dir, FLAGS.saved_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  flags.mark_flag_as_required('saved_dir')
  app.run(main)

