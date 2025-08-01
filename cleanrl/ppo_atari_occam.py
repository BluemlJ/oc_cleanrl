# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import sys
import tyro
import time
import random
import warnings

import numpy as np

from tqdm import tqdm
from rtpt import RTPT
from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from typing import Literal

import ocatari_wrappers

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
    obs_mode: Literal[
        "dqn", "obj", "occam_binary", "occam_objects",
        "occam_planes", "occam_classes"
    ] = "dqn"
    """observation mode for OCAtari"""
    buffer_window_size: int = 4
    """length of history in the observations"""
    backend: Literal["OCAtari", "HackAtari", "Gym"] = "OCAtari"
    """Which Backend should we use"""
    modifs: str = ""
    """Modifications for HackAtari"""
    new_rf: str = ""
    """Path to a new reward functions for HackAtari"""
    frameskip: int = 4
    """the frame skipping option of the environment"""

    # Tracking (Logging and monitoring configurations)
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "OCCAM"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_dir: str = None
    """the wandb directory"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    ckpt: str = ""
    """Path to a checkpoint to a model to start training from"""
    logging_level: int = 40
    """Logging level for the Gymnasium logger"""
    author: str = "JD"
    """Initials of the author"""
    checkpoint_interval: int = 40
    """Number of iterations before a model checkpoint is saved and uploaded to wandb"""

    # Algorithm-specific arguments
    architecture: Literal["PPO", "PPO_OBJ"] = "PPO"
    """the architecture to use"""

    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PPO_OBJ network parameters
    encoder_dims: list[int] = (256, 512, 1024, 512)
    """layer dimensions before nn.Flatten()"""
    decoder_dims: list[int] = (512,)
    """layer dimensions after nn.Flatten()"""

    # HackAtari testing
    test_modifs: str = ""
    """modifications for HackAtari"""

    # Imperfect detection
    detection_failure_probability: float = 0.0
    """probability that an object is not detected"""
    mislabeling_probability: float = 0.0
    """probability that an object is labeled incorrectly"""
    noise_std: float = 0.0
    """noise added to position and dimensions of objects"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed at runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed at runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed at runtime)"""
    masked_wrapper: str = None
    """the obs_mode if a masking wrapper is needed (set at runtime)"""
    add_pixels: bool = False
    """should the grayscale game screen be added to the observations (set at runtime)"""


# Global variable to hold parsed arguments
global args


# Function to create a gym environment with the specified settings
def make_env(env_id, idx, capture_video, run_dir):
    """
    Creates a gym environment with the specified settings.
    """

    def thunk():
        logger.set_level(args.logging_level)
        # Setup environment based on backend type (HackAtari, OCAtari, Gym)
        if args.backend == "HackAtari":
            from HackAtari.core import HackAtari
            modifs = [i for i in args.modifs.split(" ") if i]
            env = HackAtari(
                env_id,
                modifs=modifs,
                rewardfunc_path=args.new_rf,
                obs_mode=args.obs_mode,
                hud=False,
                render_mode="rgb_array",
                frameskip=args.frameskip,
                create_buffer_stacks=[]
            )
        elif args.backend == "OCAtari":
            from ocatari.core import OCAtari
            env = OCAtari(
                env_id,
                hud=False,
                render_mode="rgb_array",
                obs_mode=args.obs_mode,
                frameskip=args.frameskip,
                create_buffer_stacks=[]
            )
        elif args.backend == "Gym":
            # Use Gym backend with image preprocessing wrappers
            env = gym.make(env_id, render_mode="rgb_array", frameskip=args.frameskip)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, args.buffer_window_size)
        else:
            raise ValueError("Unknown Backend")

        # Capture video if required
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env,
                                           f"{run_dir}/media/videos",
                                           disable_logger=True)

        # Apply standard Atari environment wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        # Add failures to test robustness
        if (args.detection_failure_probability > 0
                or args.mislabeling_probability > 0
                or args.noise_std > 0):

            env = ocatari_wrappers.ImperfectDetectionWrapper(
                env,
                args.detection_failure_probability,
                args.mislabeling_probability,
                args.noise_std
            )

        # If masked obs_mode are set, apply correct wrapper
        if args.masked_wrapper == "occam_binary":
            env = ocatari_wrappers.BinaryMaskWrapper(env, buffer_window_size=args.buffer_window_size,
                                                     include_pixels=args.add_pixels)
        elif args.masked_wrapper == "occam_objects":
            env = ocatari_wrappers.PixelMaskWrapper(env, buffer_window_size=args.buffer_window_size,
                                                    include_pixels=args.add_pixels)
        elif args.masked_wrapper == "occam_classes":
            env = ocatari_wrappers.ObjectTypeMaskWrapper(env, buffer_window_size=args.buffer_window_size,
                                                         include_pixels=args.add_pixels)
        elif args.masked_wrapper == "occam_planes":
            env = ocatari_wrappers.ObjectTypeMaskPlanesWrapper(env, buffer_window_size=args.buffer_window_size,
                                                               include_pixels=args.add_pixels)
        elif args.masked_wrapper is not None:
            raise NotImplementedError(f"OCCAM observation mode {args.masked_wrapper} not supported!")

        return env

    return thunk


if __name__ == "__main__":
    # Parse command-line arguments using Tyro
    args = tyro.cli(Args)
    # Compute runtime-dependent arguments
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # Generate run name based on environment, experiment, seed, and timestamp
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    assert args.obs_mode != "obj" or args.architecture == "PPO_OBJ", '"obj" observations only work with "PPO_OBJ" architecture!'

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
        writer_dir = f"{args.wandb_dir}/runs/{run_name}"
        postfix = None

    # Initialize Tensorboard SummaryWriter to log metrics and hyperparameters
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # prepare for masking wrappers
    if "occam" in args.obs_mode:
        if args.obs_mode.endswith("+pixels"):
            args.masked_wrapper = args.obs_mode[:-7]
            args.add_pixels = True
        else:
            args.masked_wrapper = args.obs_mode
            args.add_pixels = False
        args.obs_mode = "ori"

    # Create RTPT object to monitor progress with estimated time remaining
    rtpt = RTPT(name_initials=args.author, experiment_name='OCALM',
                max_iterations=args.num_iterations)
    rtpt.start()  # Start RTPT tracking

    # Set logger level and determine whether to use GPU or CPU for computation
    logger.set_level(args.logging_level)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.debug(f"Using device {device}.")

    # Environment setup
    envs = SubprocVecEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, writer_dir
            ) for i in range(0, args.num_envs)
        ]
    )
    envs = VecNormalize(envs, norm_obs=False, norm_reward=True)

    # Seeding the environment and PyTorch for reproducibility
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.use_deterministic_algorithms(args.torch_deterministic)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed, args.cuda)
    envs.seed(args.seed)
    envs.action_space.seed(args.seed)

    # Define the agent's architecture based on command line arguments
    if args.architecture == "PPO":
        from architectures.ppo import PPODefault as Agent

        agent = Agent(envs, device).to(device)
    elif args.architecture == "PPO_OBJ":
        from architectures.ppo import PPObj as Agent

        agent = Agent(envs, device, args.encoder_dims, args.decoder_dims).to(device)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} does not exist!")

    # Initialize optimizer for training
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Allocate storage for observations, actions, rewards, etc.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training loop
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Iterate through training iterations with progress bar
    pbar = tqdm(range(1, args.num_iterations + 1), postfix=postfix)
    for iteration in pbar:  # Anneal learning rate if specified
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        elength = 0
        eorgr = 0
        enewr = 0
        count = 0
        done_in_episode = False

        if iteration % args.checkpoint_interval == 0:
            # Save the trained model to disk
            model_path = f"{writer_dir}/{args.exp_name}_{iteration}.cleanrl_model"
            model_data = {
                "model_weights": agent.state_dict(),
                "args": vars(args),
                "Timesteps": iteration * args.batch_size
            }
            torch.save(model_data, model_path)
            logger.info(f"model saved to {model_path} in epoch {epoch}")

            # Log model with Weights and Biases if enabled
            if args.track:
                name = f"{args.exp_name}_s{args.seed}"
                run.log_model(model_path, name)  # noqa: cannot be undefined

        # Perform rollout in each environment
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Get action and value from agent
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and store reward, next observation, and done flag
            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
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
                            eorgr += info["episode"]["r"]
                        elength += info["episode"]["l"]

        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch for optimization
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approximate KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # Normalize advantages
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss (for exploration)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backpropagation and optimizer step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Compute explained variance (diagnostic measure for value function fit quality)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log episode statistics for Tensorboard
        if done_in_episode:
            if args.new_rf:
                writer.add_scalar("charts/Episodic_New_Reward", enewr / count, global_step)
            writer.add_scalar("charts/Episodic_Original_Reward", eorgr / count, global_step)
            writer.add_scalar("charts/Episodic_Length", elength / count, global_step)
            pbar.set_description(f"Reward: {eorgr.item() / count:.1f}")

        # Log other statistics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Update RTPT for progress tracking
        rtpt.step()

    # Save the trained model to disk
    model_path = f"{writer_dir}/{args.exp_name}_final.cleanrl_model"
    model_data = {
        "model_weights": agent.state_dict(),
        "args": vars(args),
    }
    torch.save(model_data, model_path)
    logger.info(f"model saved to {model_path} in epoch {epoch}")

    # Log final model and performance with Weights and Biases if enabled
    if args.track:
        # Log model to Weights and Biases
        name = f"{args.exp_name}_s{args.seed}"
        run.log_model(model_path, name)  # noqa: cannot be undefined

        # Evaluate agent's performance
        args.new_rf = ""
        rewards = evaluate(agent, make_env, 10,
                           env_id=args.env_id,
                           capture_video=args.capture_video,
                           run_dir=writer_dir,
                           device=device)

        wandb.log({"FinalReward": np.mean(rewards)})

        if args.test_modifs != "":
            args.modifs = args.test_modifs
            args.backend = "HackAtari"
            rewards = evaluate(agent, make_env, 10,
                               env_id=args.env_id,
                               capture_video=args.capture_video,
                               run_dir=writer_dir,
                               device=device)

            wandb.log({"HackAtariReward": np.mean(rewards)})

        # Log video of agent's performance
        if args.capture_video:
            import glob

            list_of_videos = glob.glob(f"{writer_dir}/media/videos/*.mp4")
            latest_video = max(list_of_videos, key=os.path.getctime)
            wandb.log({"video": wandb.Video(latest_video)})

        wandb.finish()

    # Close environments and writer after training is complete
    envs.close()
    writer.close()