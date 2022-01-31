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
from ray.rllib.agents.ppo import ppo


from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy #0.8.5


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
    default='/home/aizaz/ray_results/PPO_Training/MA-Inde-PPO-SSUI3CCARLA/PPO_HomoNcomIndePOIntrxMASS3CTWN3-v0_0_2021-09-05_15-03-060dzvgkf4/checkpoint_100/checkpoint-100',
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
env_name_3 = "UrbanPPOTraining-v0"
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
    env = gym.make("UrbanPPOTraining-v0")

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
    ray.init(redis_address=args.redis_address,object_store_memory=10**9)
else:
    ray.init(num_gpus=args.num_gpus,object_store_memory=10**9)

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


    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE(lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.3,
    # Size of batches collected from each worker.
    # "rollout_fragment_length": 128,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    # "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 64,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 8,
    # Stepsize of SGD.
    "lr": 5e-5,
    # Learning rate schedule.
    # "lr_schedule": None,
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    "vf_share_layers": False,
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer.
    "entropy_coeff": 0.1,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.03,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    "simple_optimizer": False,
    # Use PyTorch as framework?
    "use_pytorch": False,

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
    "monitor": False,




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
    return (PPOTFPolicy, Box(0.0, 255.0, shape=(84, 84, 3)), Discrete(9),config)

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

trainer = ppo.PPOTrainer(
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

agents_reward_dict = {'car1PPO': 0.0,'car2PPO': 0.0,'car3PPO': 0.0}
done = False
i=0
action = {}
with open("info_car2PPO.json", "w") as f1:
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
        reward["car1PPO"] = 0.0
        reward["car3PPO"] = 0.0
        step += 1
        # sum up reward for all agents
        step_reward=0
        step_reward += reward["car2PPO"]
        # step_reward = step_reward-reward["car1"]
        # print ("Step reward : ",step_reward)
        writer.add_scalar("/step_reward_r", step_reward , step) 
        for agent_id in reward:
            agents_reward_dict[agent_id] += reward[agent_id]
            writer.add_scalar(agent_id + "/step_r",
                                  reward[agent_id], step)
               
            
            if agent_id=="car2PPO":
                json.dump(info[agent_id], f1)
                f1.write("\n")
                                               
            # if agent_id=="car3":
            #     json.dump(info[agent_id], f3)
            #     f3.write("\n")


        episode_reward += reward["car2PPO"]
        # episode_reward = episode_reward-reward["car1"]
        # print ("Episode reward : ",episode_reward)
        writer.add_scalar("/episode_reward_r", episode_reward , step) 
        pprint (" --- ")




print (" ====================== ======================= ")    


done = done['__all__']
env_3.close()
writer.close()


ray.shutdown()