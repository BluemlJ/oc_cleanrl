# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import warnings
import os
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path

import tyro
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from rtpt import RTPT
import gymnasium as gym
from gymnasium import logger

# Suppress warnings to avoid cluttering output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set CUDA environment variable for determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add custom paths if OC_ATARI_DIR is set (optional integration for extended functionality)
oc_atari_dir = os.getenv("OC_ATARI_DIR")
if oc_atari_dir is not None:
    oc_atari_path = os.path.join(Path(__file__), oc_atari_dir)
    sys.path.insert(1, oc_atari_path)

# Add the evaluation directory to the Python path to import custom evaluation functions
eval_dir = os.path.join(Path(__file__).parent.parent, "cleanrl_utils/evals/")
sys.path.insert(1, eval_dir)
from generic_eval import evaluate  # noqa

# Command line argument configuration using dataclass
@dataclass
class Args:
    # General
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Environment parameters
    env_id: str = "ALE/Pong-v5"
    """the id of the environment"""
    obs_mode: str = "dqn"
    """observation mode for OCAtari"""
    feature_func: str = ""
    """the object features to use as observations"""
    buffer_window_size: int = 4
    """length of history in the observations"""
    backend: str = "OCAtari"
    """Which Backend should we use"""
    modifs: str = ""
    """Modifications for Hackatari"""
    new_rf: str = ""
    """Path to a new reward functions for OCALM and HACKATARI"""
    frameskip: int = 4
    """the frame skipping option of the environment"""

    # Tracking (Logging and monitoring configurations)
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "OC-Transformer"
    """the wandb's project name"""
    wandb_entity: str = "AIML_OC"
    """the entity (team) of wandb's project"""
    wandb_dir: str = None
    """the wandb directory"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    ckpt: str = ""
    """Path to a checkpoint to a model to start training from"""
    logging_level: int = 40
    """Logging level for the Gymnasium logger"""
    author : str = "JB"
    """Initials of the author"""

        # Algorithm-specific arguments
    architecture: str = "DQN"
    """Specifies the used architecture"""

    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 10000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

# Global variable to hold parsed arguments
global args


# Function to create a gym environment with the specified settings
def make_env(env_id, idx, capture_video, run_dir, feature_func="xywh", window_size=4, frameskip=4):
    """
    Creates a gym environment with the specified settings.
    """

    def thunk():
        logger.set_level(args.logging_level)
        # Setup environment based on backend type (HackAtari, OCAtari, Gym)
        if args.backend == "HackAtari":
            from hackatari.core import HackAtari
            modifs = [i for i in args.modifs.split(" ") if i]
            env = HackAtari(
                env_id,
                modifs=modifs,
                rewardfunc_path=args.new_rf,
                obs_mode=args.obs_mode,
                hud=False,
                render_mode="rgb_array",
                frameskip=args.frameskip
            )
        elif args.backend == "OCAtari":
            from ocatari.core import OCAtari
            env = OCAtari(
                env_id,
                hud=False,
                render_mode="rgb_array",
                obs_mode=args.obs_mode,
                frameskip=args.frameskip
            )
        elif args.backend == "Gym":
            # Use Gym backend with image preprocessing wrappers
            env = gym.make(env_id, render_mode="rgb_array", frameskip=args.frameskip)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        else:
            raise ValueError("Unknown Backend")

        # Capture video if required
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"{run_dir}/media/videos", disable_logger=True)

        # Apply standard Atari environment wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
            
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
from architectures.dqn import QNetwork_C51 as QNetwork

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
    
if __name__ == "__main__":
    
    # Parse command-line arguments using Tyro
    args = tyro.cli(Args)
    # Generate run name based on environment, experiment, seed, and timestamp
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"  

    # Create RTPT object to monitor progress with estimated time remaining
    rtpt = RTPT(
        name_initials=args.author, experiment_name="OCALM",
        max_iterations=args.total_timesteps
    )
    rtpt.start()  # Start RTPT tracking

    # Initialize tracking with Weights and Biases if enabled
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.wandb_dir
        )
        writer_dir = run.dir
        postfix = dict(url=run.url)
    else:
        writer_dir = f"{args.wandb_dir}/{run_name}"
        postfix = None

    # Set logger level and determine whether to use GPU or CPU for computation
    logger.set_level(args.logging_level)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.debug(f"Using device {device}.")

    # env setup
    #envs = gym.vector.SyncVectorEnv(
    #    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    #)
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Environment setup
    envs = SubprocVecEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, writer_dir,
                args.feature_func, args.buffer_window_size, args.frameskip
            ) for i in range(0, args.num_envs)
        ]
    )
    envs = VecNormalize(envs, norm_obs=False, norm_reward=True)

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Seeding the environment and PyTorch for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed, args.cuda)
    envs.seed(args.seed)
    envs.action_space.seed(args.seed)
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    elength = 0
    eorgr = 0
    enewr = 0
    count = 0
    done_in_episode = False

    # TRY NOT TO MODIFY: start the game
    #import ipdb;ipdb.set_trace()
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = envs.step(actions)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        
        # Track episode-level statistics if a game is done
        if 1 in next_done:
            for info in infos:
                if "episode" in info:
                    count += 1
                    done_in_episode = True
                    if args.new_rf:
                        enewr += info["episode"]["r"]
                        eorgr += info["org_return"]
                    else:
                        eorgr += info["episode"]["r"].item()
                    elength += info["episode"]["l"]

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        rb.add(obs, next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                #import ipdb; ipdb.set_trace()
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data.next_observations)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

        # Log episode statistics for Tensorboard
        if done_in_episode:
            if args.new_rf:
                writer.add_scalar("charts/Episodic_New_Reward", enewr / count, global_step)
            writer.add_scalar("charts/Episodic_Original_Reward", eorgr / count, global_step)
            writer.add_scalar("charts/Episodic_Length", elength / count, global_step)
            pbar.set_description(f"Reward: {eorgr / count:.1f}")
            elength = 0
            eorgr = 0
            enewr = 0
            count = 0
            done_in_episode = False

        # Update RTPT for progress tracking
        rtpt.step()


    # Save the trained model to disk
    model_path = f"{writer_dir}/{args.exp_name}.cleanrl_model"
    model_data = {
        "model_weights": agent.state_dict(),
        "args": vars(args),
    }
    torch.save(model_data, model_path)
    logger.info(f"model saved to {model_path} in epoch {epoch}")

    # Log final model and performance with Weights and Biases if enabled
    if args.track:
        # Evaluate agent's performance
        args.new_rf = ""
        rewards = evaluate(
            agent, make_env, 10,
            env_id=args.env_id,
            capture_video=args.capture_video,
            run_dir=writer_dir,
            feature_func=args.feature_func,
            window_size=args.buffer_window_size
        )

        wandb.log({"FinalReward": np.mean(rewards)})

        # Log model to Weights and Biases
        name = f"{args.exp_name}_s{args.seed}"
        run.log_model(model_path, name)  # noqa: cannot be undefined

        # Log video of agent's performance
        if args.capture_video:
            import glob
            list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
            latest_video = max(list_of_videos, key=os.path.getctime)
            wandb.log({"video": wandb.Video(latest_video)})

        wandb.finish()

    envs.close()
    writer.close()
