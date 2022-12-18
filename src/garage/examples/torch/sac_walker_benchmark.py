#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

import yaml
from yaml.loader import SafeLoader
import os
import argparse

# @wrap_experiment(snapshot_mode='none')
def sac_walker_batch(ctxt=None, 
    seed=1, 
    n_epochs=10, 
    batch_size=1024, 
    policy_net_size=[256, 256], 
    q_function_net_size=[256, 256], 
    layer_normalization=False, 
    activation_function="relu", 
    reward_scale=1.):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    env = normalize(GymEnv('Hopper-v2'))

    assert activation_function in ["relu", "tanh", "leakyrelu"] 
    
    if activation_function == "tanh":
        hidden_nonlinearity=F.tanh
    elif activation_function == "leakyrelu":
        hidden_nonlinearity=F.leaky_relu
    else :
        hidden_nonlinearity=F.relu

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=policy_net_size,
        hidden_nonlinearity=hidden_nonlinearity, # nn.ReLU
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
        layer_normalization=layer_normalization
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=q_function_net_size,
                                hidden_nonlinearity=hidden_nonlinearity,
                                layer_normalization=layer_normalization)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=q_function_net_size,
                                hidden_nonlinearity=hidden_nonlinearity,
                                layer_normalization=layer_normalization)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = LocalSampler(agents=policy,
                        envs=env,
                        max_episode_length=env.spec.max_episode_length,
                        worker_class=FragmentWorker)

    sac = SAC(env_spec=env.spec,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            sampler=sampler,
            gradient_steps_per_itr=1000,
            max_episode_length_eval=1000,
            replay_buffer=replay_buffer,
            min_buffer_size=1e4,
            target_update_tau=5e-3,
            discount=0.99,
            buffer_batch_size=256,
            reward_scale=reward_scale,
            steps_per_epoch=1,
            use_deterministic_evaluation=True,
            fixed_alpha=None)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=n_epochs, batch_size=batch_size)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparameter_to_analyze", 
        help="hyperparameter to analyze", 
        required=True, 
        choices=["batch_size", "layer_normalization", "activation_functions", "policy_net", "q_function_net", "reward_scale"]
    )

    parser.add_argument(
        "--continue_training", 
        help="continue stopped training", 
        action="store_true",
    )
    parser.set_defaults(continue_training=False)

    args = parser.parse_args()
    hyperparameter_to_analyze = args.hyperparameter_to_analyze
    continue_training = args.continue_training

    benchmark_config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_config.yml", 
    )

    # Open the file and load the file
    with open(benchmark_config_file) as f:
        benchmark_config_data = yaml.load(f, Loader=SafeLoader)


    algo = benchmark_config_data["algo"]
    env = benchmark_config_data["env"]
    seeds = benchmark_config_data["seeds"]
    epochs = benchmark_config_data["epochs"]
    combinations_to_skip = benchmark_config_data["combinations_to_skip"]

    batch_size_list = benchmark_config_data["batch_size_list"]
    policy_net_sizes = benchmark_config_data["network_size"]["policy_net"]
    q_function_net_sizes = benchmark_config_data["network_size"]["q_function_net"]
    layer_normalization_choices = benchmark_config_data["layer_normalization"]
    activation_functions_choices = benchmark_config_data["activation_functions"]
    reward_scale_choices = benchmark_config_data["reward_scale"]

    for seed in seeds:
        if hyperparameter_to_analyze == "batch_size":
            for batch_size in  batch_size_list:
                if [seed, batch_size] in combinations_to_skip and continue_training:
                    continue
                log_dirname = f"{algo}_{hyperparameter_to_analyze}_{str(batch_size)}_{seed}"
                wrap_experiment(snapshot_mode='none', name=log_dirname)(sac_walker_batch)(seed=seed, n_epochs=epochs, batch_size=batch_size)

        elif hyperparameter_to_analyze == "layer_normalization":
            for layer_normalization in  layer_normalization_choices:
                if [seed, layer_normalization] in combinations_to_skip and continue_training:
                    continue
                log_dirname = f"{algo}_{hyperparameter_to_analyze}_{str(layer_normalization)}_{seed}"
                wrap_experiment(snapshot_mode='none', name=log_dirname)(sac_walker_batch)(seed=seed, n_epochs=epochs, layer_normalization=layer_normalization)

        elif hyperparameter_to_analyze == "activation_functions":
            for activation_function in  activation_functions_choices:
                log_dirname = f"{algo}_{hyperparameter_to_analyze}_{str(activation_function)}_{seed}"
                wrap_experiment(snapshot_mode='none', name=log_dirname)(sac_walker_batch)(seed=seed, n_epochs=epochs, activation_function=activation_function)

        elif hyperparameter_to_analyze == "policy_net":
            for policy_net_size in  policy_net_sizes:
                log_dirname = f"{algo}_{hyperparameter_to_analyze}_{str(policy_net_size)}_{seed}"
                wrap_experiment(snapshot_mode='none', name=log_dirname)(sac_walker_batch)(seed=seed, n_epochs=epochs, policy_net_size=policy_net_size)

        elif hyperparameter_to_analyze == "q_function_net":
            for q_function_net_size in  q_function_net_sizes:
                log_dirname = f"{algo}_{hyperparameter_to_analyze}_{str(q_function_net_size)}_{seed}"
                wrap_experiment(snapshot_mode='none', name=log_dirname)(sac_walker_batch)(seed=seed, n_epochs=epochs, q_function_net_size=q_function_net_size)

        elif hyperparameter_to_analyze == "reward_scale":
            for reward_scale in  reward_scale_choices:
                log_dirname = f"{algo}_{hyperparameter_to_analyze}_{str(reward_scale)}_{seed}"
                wrap_experiment(snapshot_mode='none', name=log_dirname)(sac_walker_batch)(seed=seed, n_epochs=epochs, reward_scale=reward_scale)
        
        else:
            print("wrong choice")

if __name__ == "__main__":
    main()