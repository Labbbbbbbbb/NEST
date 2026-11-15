# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from copy import deepcopy
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
        self.actions = []
        self.gamefile = None
        self.initial_observation = None
        self.current_observation = None
        self.domain_randomization = bool(config["env"]["domain_randomization"])
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        self.actions.append(action)
        self.current_observation = obs[0]
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        self.actions = []
        self.gamefile = infos.get("extra.gamefile", [None])[0]
        self.initial_observation = self.current_observation = obs[0]
        return obs, infos

    def snapshot(self):
        if self.domain_randomization:
            raise ValueError("ARPO deterministic replay requires domain_randomization=False")
        if not self.gamefile:
            raise RuntimeError("Cannot fork without the current ALFWorld game file")
        return {"gamefile": self.gamefile, "actions": list(self.actions),
                "initial_observation": self.initial_observation,
                "current_observation": self.current_observation}

    def restore(self, state):
        """Replay a prefix without consuming the dataset's next-game iterator.

        TextWorld's batched Gym environment chooses its game via this iterator.
        Temporarily selecting the source game preserves the destination worker's
        normal reset schedule, including after multiple forks and future epochs.
        """
        if self.domain_randomization:
            raise ValueError("ARPO deterministic replay requires domain_randomization=False")
        base = getattr(self.env, "unwrapped", self.env)
        iterator = base._gamefiles_iterator
        try:
            base._gamefiles_iterator = iter([state["gamefile"]])
            obs, infos = self.reset()
        finally:
            base._gamefiles_iterator = iterator
        if obs[0] != state["initial_observation"]:
            raise RuntimeError("ARPO replay changed the initial observation")
        for action in state["actions"]:
            obs, _, _, infos = self.step(action)
        if obs[0] != state["current_observation"]:
            raise RuntimeError("ARPO replay failed to reproduce the source observation")
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]
        self.active_masks = None
        self.last_results = [None] * self.num_processes

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        active_indices = [i for i in range(self.num_processes)
                          if self.active_masks is None or self.active_masks[i]]
        for i in active_indices:
            futures.append(self.workers[i].step.remote(actions[i]))

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        for i, result in zip(active_indices, ray.get(futures)):
            self.last_results[i] = result
        results = deepcopy(self.last_results)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            self.last_results[i] = (deepcopy(obs), [0.0], [False], deepcopy(info))
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def fork(self, source, target):
        if self.multi_modal:
            raise ValueError("ARPO replay only supports text ALFWorld")
        if source // self.group_n != target // self.group_n:
            raise ValueError("ARPO may only fork within the same task group")
        state = ray.get(self.workers[source].snapshot.remote())
        obs, info = ray.get(self.workers[target].restore.remote(state))
        self.prev_admissible_commands[target] = deepcopy(info['admissible_commands'][0])
        self.last_results[target] = (deepcopy(obs), [0.0], [False], deepcopy(info))

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)