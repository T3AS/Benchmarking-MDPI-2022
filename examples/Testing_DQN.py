#!/bin/env python
import gym
import macad_gym  # noqa F401
import argparse
import os
from pprint import pprint

import cv2
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete
from macad_agents.rllib.env_wrappers import wrap_deepmind
from macad_agents.rllib.models import register_mnih15_net

import datetime
import json
from ray.rllib.agents.dqn import dqn


from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy #0.8.5


from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import register_env
import time
from pprint import pprint
import pickle
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorboardX import SummaryWriter
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
writer = SummaryWriter("logss/" + timestamp)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
# tf.keras.backend.set_session(tf.Session(config=config));

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

from ray.rllib.agents.trainer_template import build_trainer



parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    default="PongNoFrameskip-v4",
    help="Name Gym env. Used only in debug mode. Default=PongNoFrameskip-v4")
parser.add_argument(
    "--checkpoint-path",
    #Replace it with your path of last training checkpoints
    default='/home/aizaz/ray_results/DQN_Training/MA-Inde-DQN-SSUI3CCARLA/DQN_HomoNcomIndePOIntrxMASS3CTWN3-v0_0_2021-09-06_08-12-10v_gxzly8/checkpoint_100/checkpoint-100',
    help="Path to checkpoint to resume training")
# parser.add_argument(
#     "--checkpoint-path2",
#     #Replace it with your path of last training checkpoints
#     default='/home/aizaz/ray_results/A2C_Training/MA-Inde-A2C-SSUI3CCARLA/A2C_HomoNcomIndePOIntrxMASS3CTWN3-v0_0_2021-09-08_07-28-331qh6yk0w/checkpoint_100/checkpoint-100',
#     help="Path to checkpoint to resume training")    
parser.add_argument(
    "--disable-comet",
    action="store_true",
    help="Disables comet logging. Used for local smoke tests")
parser.add_argument(
    "--num-workers",
    default=1, #2
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=1, type=int, help="Number of gpus to use. Default=2")
parser.add_argument(
    "--sample-bs-per-worker",
    default=1024,
    type=int,
    help="Number of samples in a batch per worker. Default=50")
parser.add_argument(
    "--train-bs",
    default=128,
    type=int,
    help="Train batch size. Use as per available GPU mem. Default=500")
parser.add_argument(
    "--envs-per-worker",
    default=1,
    type=int,
    help="Number of env instances per worker. Default=10")
parser.add_argument(
    "--notes",
    default=None,
    help="Custom experiment description to be added to comet logs")
parser.add_argument(
    "--model-arch",
    default="mnih15",
    help="Model architecture to use. Default=mnih15")
parser.add_argument(
    "--num-steps",
    default=2000000, 
    type=int,
    help="Number of steps to train. Default=20M")
parser.add_argument(
    "--num-iters",
    default=1, #20
    type=int,
    help="Number of training iterations. Default=20")
parser.add_argument(
    "--log-graph",
    action="store_true",
    help="Write TF graph on Tensorboard for debugging")
parser.add_argument(
    "--num-framestack",
    type=int,
    default=4,
    help="Number of obs frames to stack")
parser.add_argument(
    "--redis-address",
    default=None,
    help="Address of ray head node. Be sure to start ray with"
    "ray start --redis-address <...> --num-gpus<.> before running this script")
parser.add_argument(
    "--use-lstm", action="store_true", help="Append a LSTM cell to the model")

args = parser.parse_args()

#--------------------------------------------------------------------
model_name = args.model_arch
if model_name == "mnih15":
    register_mnih15_net()  # Registers mnih15
else:
    print("Unsupported model arch. Using default")
    register_mnih15_net()
    model_name = "mnih15"

# # Used only in debug mode
# env_name = "UrbanPPOTraining-v0"
# env = gym.make(env_name)
# env_actor_configs = env.configs
# num_framestack = args.num_framestack
# # env_config["env"]["render"] = False

# # Used only in debug mode
# env_name_2 = "UrbanA2CTraining-v0"
# env_2 = gym.make(env_name_2)
# env_actor_configs_2 = env_2.configs
# num_framestack = args.num_framestack
# # env_config["env"]["render"] = False

# Used only in debug mode
env_name_3 = "UrbanDQNTraining-v0"
env_3 = gym.make(env_name_3)
env_actor_configs_3 = env_3.configs
num_framestack = args.num_framestack
# env_config["env"]["render"] = False

#--------------------------------------------------------------------

# def env_creator(env_config):
#     # NOTES: env_config.worker_index & vector_index are useful for
#     # curriculum learning or joint training experiments
#     import macad_gym
#     env = gym.make("UrbanPPOTraining-v0")

#     # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
#     # stack frames & some more op
#     env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
#     return env

# def env_creator_2(env_config):
#     # NOTES: env_config.worker_index & vector_index are useful for
#     # curriculum learning or joint training experiments
#     import macad_gym
#     env = gym.make("UrbanA2CTraining-v0")

#     # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
#     # stack frames & some more op
#     env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
#     return env

def env_creator_3(env_config):
    # NOTES: env_config.worker_index & vector_index are useful for
    # curriculum learning or joint training experiments
    import macad_gym
    env = gym.make("UrbanDQNTraining-v0")

    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env
# register_env(env_name, lambda config: env_creator(config))
# register_env(env_name_2, lambda config: env_creator_2(config))
register_env(env_name_3, lambda config: env_creator_3(config))

#--------------------------------------------------------------------

# Placeholder to enable use of a custom pre-processor
class ImagePreproc(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)  # Adjust third dim if stacking frames
        return self.shape

    def transform(self, observation):
        observation = cv2.resize(observation, (self.shape[0], self.shape[1]))
        return observation
def transform(self, observation):
        observation = cv2.resize(observation, (self.shape[0], self.shape[1]))
        return observation

ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)
#--------------------------------------------------------------------

if args.redis_address is not None:
    # num_gpus (& num_cpus) must not be provided when connecting to an
    # existing cluster
    ray.init(redis_address=args.redis_address)#,object_store_memory=10**9)
else:
    ray.init(num_gpus=args.num_gpus)#,object_store_memory=10**9)

config = {
    # Model and preprocessor options.
    "model": {
        "custom_model": model_name,
        "custom_options": {
            # Custom notes for the experiment
            "notes": {
                "args": vars(args)
            },
        },
        # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
        "custom_preprocessor": "sq_im_84",
        "dim": 84,
        "free_log_std": False,  # if args.discrete_actions else True,
        "grayscale": True,
        # conv_filters to be used with the custom CNN model.
        # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2], [16, [3, 3], 2]]
    },
    # preproc_pref is ignored if custom_preproc is specified
    # "preprocessor_pref": "deepmind",

    # env_config to be passed to env_creator
    
    "env_config": env_actor_configs_3
}

def default_policy():
    env_actor_configs_3["env"]["render"] = False

    config = {
    # Model and preprocessor options.
    "model": {
        "custom_model": model_name,
        "custom_options": {
            # Custom notes for the experiment
            "notes": {
                "args": vars(args)
            },
        },
        # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
        "custom_preprocessor": "sq_im_84",
        "dim": 84,
        "free_log_std": False,  # if args.discrete_actions else True,
        "grayscale": True,
        # conv_filters to be used with the custom CNN model.
        # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2], [16, [3, 3], 2]]
    },


   # === Model ===
    # Number of atoms for representing the distribution of return. When
    # this is greater than 1, distributional Q-learning is used.
    # the discrete supports are bounded by v_min and v_max
    "num_atoms": 1,
    "v_min": -10.0,
    "v_max": 10.0,
    # Whether to use noisy network
    "noisy": False,
    # control the initial value of noisy nets
    "sigma0": 0.5,
    # Whether to use dueling dqn
    "dueling": True,
    # Dense-layer setup for each the advantage branch and the value branch
    # in a dueling architecture.
    "hiddens": [256],
    # Whether to use double dqn
    "double_q": True,
    # N-step Q learning
    "n_step": 1,

    # === Exploration Settings ===
    "exploration_config": {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

        # For soft_q, use:
        # "exploration_config" = {
        #   "type": "SoftQ"
        #   "temperature": [float, e.g. 1.0]
        # }
    },
    # Switch to greedy actions in evaluation workers.
    "evaluation_config": {
        "explore": False,
    },

    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of iterations.
    "timesteps_per_iteration": 100,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 500,
    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": 500,
    # The number of contiguous environment steps to replay at once. This may
    # be set to greater than 1 to support recurrent models.
    "replay_sequence_length": 1,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Final value of beta (by default, we use constant beta=0.4).
    "final_prioritized_replay_beta": 0.4,
    # Time steps over which the beta parameter is annealed.
    "prioritized_replay_beta_annealing_timesteps": 20000,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,

    # Whether to LZ4 compress observations
    "compress_observations": False,
    # Callback to run before learning on a multi-agent batch of experiences.
    "before_learn_on_batch": None,

    # The intensity with which to update the model (vs collecting samples from
    # the env). If None, uses the "natural" value of:
    # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
    # `num_envs_per_worker`).
    # If provided, will make sure that the ratio between ts inserted into and
    # sampled from the buffer matches the given value.
    # Example:
    #   training_intensity=1000.0
    #   train_batch_size=250 rollout_fragment_length=1
    #   num_workers=1 (or 0) num_envs_per_worker=1
    #   -> natural value = 250 / 1 = 250.0
    #   -> will make sure that replay+train op will be executed 4x as
    #      often as rollout+insert op (4 * 250 = 1000).
    # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further details.
    "training_intensity": None,

    # === Optimization ===
    # Learning rate for adam optimizer
    "lr": 5e-4,
    # Learning rate schedule
    "lr_schedule": None,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    # If not None, clip gradients during optimization at this value
    "grad_clip": 40,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1000,
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 4,
    # Size of a batch sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 32,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,

    # Discount factor of the MDP.
    "gamma": 0.99,
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": 1024,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": True,
    # Don't set 'done' at the end of the episode. Note that you still need to
    # set this if soft_horizon=True, unless your env is actually running
    # forever without returning done=True.
    "no_done_at_end": True,
    "monitor": True,




    # System params.
    # Should be divisible by num_envs_per_worker
    "sample_batch_size":
     args.sample_bs_per_worker,
    "train_batch_size":
    args.train_bs,
    # "rollout_fragment_length": 128,
    "num_workers":
    args.num_workers,
    # Number of environments to evaluate vectorwise per worker.
    "num_envs_per_worker":
    args.envs_per_worker,
    "num_cpus_per_worker":
    1,
    "num_gpus_per_worker":
    1,
    # "eager_tracing": True,

    # # Learning params.
    # "grad_clip":
    # 40.0,
    # "clip_rewards":
    # True,
    # either "adam" or "rmsprop"
    "opt_type":
    "adam",
    # "lr":
    # 0.003,
    "lr_schedule": [
        [0, 0.0006],
        [20000000, 0.000000000001],  # Anneal linearly to 0 from start 2 end
    ],
    # rmsprop considered
    "decay":
    0.5,
    "momentum":
    0.0,
    "epsilon":
    0.1,
    # # balancing the three losses
    # "vf_loss_coeff":
    # 0.5,  # Baseline loss scaling
    # "entropy_coeff":
    # -0.01,

    # preproc_pref is ignored if custom_preproc is specified
    # "preprocessor_pref": "deepmind",
   # "gamma": 0.99,

    "use_lstm": args.use_lstm,
    # env_config to be passed to env_creator
    "env":{
        "render": True
    },
    # "in_evaluation": True,
    # "evaluation_num_episodes": 1,
    "env_config": env_actor_configs_3
    }






    # pprint (config)
    return (DQNTFPolicy, Box(0.0, 255.0, shape=(84, 84, 3)), Discrete(9),config)

def update_checkpoint_for_rollout(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        extra_data = pickle.load(f)
    if not "trainer_state" in extra_data:
        extra_data["trainer_state"] = {}
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(extra_data, f)

update_checkpoint_for_rollout(args.checkpoint_path)
# update_checkpoint_for_rollout(args.checkpoint_path2)

pprint (args.checkpoint_path)
pprint(os.path.isfile(args.checkpoint_path))

multiagent = True

trainer = dqn.DQNTrainer(
    env=env_name_3,
    # Use independent policy graphs for each agent
    config={

        "multiagent": {
            "policies": {
                id: default_policy()
                for id in env_actor_configs_3["actors"].keys()
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "env_config": env_actor_configs_3,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.envs_per_worker,
        "sample_batch_size": args.sample_bs_per_worker,
        # "rollout_fragment_length": args.sample_bs_per_worker,

        "train_batch_size": args.train_bs,
 
    })
# Restore all policies from checkpoint.
trainer.restore(args.checkpoint_path)

# # Get untrained weights for all policies.
# untrained_weights = trainer.get_weights()
# # Restore all policies from checkpoint.
# trainer.restore(args.checkpoint_path)
# # Set back all weights (except for 1st agent) to original
# # untrained weights.
# trainer.set_weights(
#     {pid: w
#      for pid, w in untrained_weights.items() if pid != "car2PPO"})




#--------------------------------------------------------------------
agents_reward_dict = {}

obs = env_3.reset()
#for ep in range(2):
step = 0


episode_reward = 0
info_dict= []

agents_reward_dict = {'car1DQN': 0.0,'car2DQN': 0.0,'car3DQN': 0.0}
done = False
i=0
action = {}
with open("info_car2DQN.json", "w") as f1:
    print ("Starting a single episode for testing")
    while i < 2000:  # number of steps in a episodic run 
        i += 1
        for agent_id, agent_obs in obs.items():
            policy_id = trainer.config["multiagent"]["policy_mapping_fn"](agent_id)
            # pprint (policy_id)
            
            action[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
            # print (action[agent_id])
            #print (" -***************-- ")
        obs, reward, done, info = env_3.step(action)
        reward["car1DQN"] = 0.0
        reward["car3DQN"] = 0.0
        step += 1
        # sum up reward for all agents
        step_reward=0
        step_reward += reward["car2DQN"]
        # step_reward = step_reward-reward["car1"]
        # print ("Step reward : ",step_reward)
        writer.add_scalar("/step_reward_r", step_reward , step) 
        for agent_id in reward:
            agents_reward_dict[agent_id] += reward[agent_id]
            writer.add_scalar(agent_id + "/step_r",
                                  reward[agent_id], step)
               
            
            if agent_id=="car2DQN":
                json.dump(info[agent_id], f1)
                f1.write("\n")
                                               
            # if agent_id=="car3":
            #     json.dump(info[agent_id], f3)
            #     f3.write("\n")


        episode_reward += reward["car2DQN"]
        # episode_reward = episode_reward-reward["car1"]
        # print ("Episode reward : ",episode_reward)
        writer.add_scalar("/episode_reward_r", episode_reward , step) 
        pprint (" --- ")




print (" ====================== ======================= ")    


done = done['__all__']
env_3.close()
writer.close()


ray.shutdown()