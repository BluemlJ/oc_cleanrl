from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    agent,
    make_env: Callable,
    eval_episodes: int,
    device,
    **env_kwargs
):
    envs = gym.vector.SyncVectorEnv([make_env(idx=0, **env_kwargs)])
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns
