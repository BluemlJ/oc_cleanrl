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
from gymnasium import logger, ObservationWrapper

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    NoopResetEnv,
)
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from architectures.common import layer_init, NormalizeImg

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
    obs_mode: str = "ori"
    """observation mode for OCAtari"""
    feature_func: str = ""
    """the object features to use as observations"""
    buffer_window_size: int = 1
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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "PPObj-v2"
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
    author: str = "CD"
    """Initials of the author"""
    checkpoint_interval: int = 40
    """Number of iterations before a model checkpoint is saved and uploaded to wandb"""

    # Algorithm-specific arguments
    architecture: str = "PPO_OBJ"
    """ Specifies the used architecture"""

    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
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

    # HackAtari testing
    test_modifs: str = ""
    """Modifications for Hackatari"""

    # PPObj network parameters
    layer_dims: tuple[int, ...] = (256, 512, 1024, 512)
    """layer dimensions for MoEAgent"""

    # Wrapper
    model_dir: str = ""
    """The model folder"""
    moe_wrappers: tuple[str, ...] = ("plane_masks",)
    """The agents/wrappers to use in the MoE model"""
    include_raw_obs: bool = True
    """Include raw observations alongside expert policies for the MoE input"""
    raw_obs_downsample: tuple[int, int] = (84, 84)
    """Target (height, width) for downsampling raw observations when included"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    masked_wrapper: str = "obj_dx"
    """the obs_mode if a masking wrapper is needed"""


agent_mapping = {
            "plane_masks": "ppo_occam_planes",
            "class_masks": "ppo_occam_classes",
            "binary_masks": "ppo_occam_binary",
            "object_masks": "ppo_occam_pixels",
            # "pixel_planes": "ppo_occam_pixel_planes",  # todo
            # "big_planes": "ppo_occam_parallel_planes",  # todo
            "dqn": "pixel_ppo",
            "obj": "obj_ppo_large",
        }


def load_agent(env, model_dir, game_name, wrapper, target_device):
    """Load a saved CleanRL PPO agent for a given OCAtari wrapper."""
    ckpt = torch.load(
        f"{model_dir}/{game_name}/0/{agent_mapping[wrapper]}.cleanrl_model",
        map_location=target_device,
        weights_only=False,
    )
    ppo_agent = PPOAgent(env, target_device).to(target_device)
    ppo_agent.load_state_dict(ckpt["model_weights"])
    ppo_agent.eval()

    return ppo_agent


class PPOAgent(nn.Module):
    """Convolutional policy used by the individual expert checkpoints."""

    def __init__(self, env, device):
        """Initialise the expert network architecture and preprocessing."""
        super().__init__()

        dims = env.observation_space.shape
        self.device = device

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(dims[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU()
        )

        self.norm_img = NormalizeImg()

        self.network.append(nn.Flatten())

        # compute flatten size with a dummy forward
        # makes the agent applicable for any input image size
        with torch.no_grad():
            f = self.network(torch.zeros((1,) + dims))
            feat_dim = f.flatten().shape[0]

        self.network.append(layer_init(nn.Linear(feat_dim, 512)))
        self.network.append(nn.ReLU())

        self.actor = layer_init(nn.Linear(512, env.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        """Return the action distribution for an individual wrapper observation."""
        x = torch.from_numpy(self.norm_img(
            x)).float().unsqueeze(0).to(self.device)
        hidden = self.network(x)
        logits = self.actor(hidden)
        return Categorical(logits=logits)


class RawObservationCache(ObservationWrapper):
    """Attach to the env to keep the most recent raw pixel observation in memory."""

    def __init__(self, env):
        super().__init__(env)
        self.last_raw_observation = None

    def observation(self, observation):
        """Store the unmodified observation and return it unchanged."""
        self.last_raw_observation = observation
        return observation


class MoEWrapper(ObservationWrapper):
    """Combine expert policy outputs (and optionally raw pixels) into a flat vector."""

    def __init__(self, env, model_dir, target_device, include_raw_obs, downsample_hw):

        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert isinstance(env.action_space, gym.spaces.Discrete)

        self.device = target_device if isinstance(
            target_device, torch.device) else torch.device(target_device)
        self.include_raw_obs = include_raw_obs
        self.target_hw = downsample_hw
        self.agents = {}
        for wrapper in env.wrappers:  # noqa: should wrap an occam multi wrapper
            self.agents[wrapper] = load_agent(env.wrappers[wrapper], model_dir, env.unwrapped.game_name, wrapper, self.device)

        super().__init__(env)

        self.num_experts = len(self.agents)
        self.action_dim = env.unwrapped.action_space.n
        self.policy_feature_dim = self.action_dim * self.num_experts

        if include_raw_obs:
            raw_space = env.unwrapped.observation_space
            if isinstance(raw_space, gym.spaces.Box):
                self.raw_obs_shape = raw_space.shape
                dummy_obs = np.zeros(self.raw_obs_shape, dtype=np.float32)
                encoded = self._encode_raw_observation(dummy_obs)
                self.raw_obs_dim = encoded.size
                self.encoded_raw_hw = self.target_hw
                self._empty_raw_flat = np.zeros(self.raw_obs_dim, dtype=np.float32)
            else:
                self.raw_obs_shape = ()
                self.raw_obs_dim = 0
                self.encoded_raw_hw = None
                self._empty_raw_flat = None
        else:
            self.raw_obs_shape = ()
            self.raw_obs_dim = 0
            self.encoded_raw_hw = None
            self._empty_raw_flat = None

        total_dim = self.policy_feature_dim + self.raw_obs_dim
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(total_dim,), dtype=np.float32,
        )

    def observation(self, observation):
        """Concatenate expert probabilities and optional raw pixels for the MoE agent."""
        with torch.no_grad():
            policy_chunks = []
            for key, agent in self.agents.items():
                policy = agent(observation[key])
                policy_chunks.append(
                    policy.probs[0].detach().cpu().numpy().astype(np.float32))

        policy_vec = np.concatenate(policy_chunks, axis=0).astype(np.float32)

        if self.raw_obs_dim == 0:
            # Raw pixels disabled: only feed expert policy probabilities downstream
            return policy_vec

        raw_obs = self.env.last_raw_observation  # todo why not use the grayscale and downsample directly?
        if raw_obs is None:
            raw_flat = self._empty_raw_flat
        else:
            raw_flat = self._encode_raw_observation(raw_obs)

        return np.concatenate((policy_vec, raw_flat), axis=0).astype(np.float32)

    def _encode_raw_observation(self, raw_obs, target_hw=None):
        """
        Down-sample and normalise the raw observation so it does not dominate the MoE input.

        Returns:
            np.ndarray: flattened low-resolution representation of the raw frame.
        """
        if target_hw is None:
            target_hw = self.target_hw

        arr = np.asarray(raw_obs, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Collapse channel / stack dimensions to a single grayscale image
        if arr.ndim == 3:
            if arr.shape[0] <= 4:
                arr = arr.mean(axis=0)  # channels-first (C,H,W)
            elif arr.shape[-1] <= 4:
                arr = arr.mean(axis=-1)  # channels-last (H,W,C)
            else:
                arr = arr.mean(axis=-1)
        elif arr.ndim > 3:
            arr = arr.mean(axis=0)
            return self._encode_raw_observation(arr, target_hw)

        if arr.ndim == 1:
            return arr
        if arr.ndim == 0:
            return np.array([arr], dtype=np.float32)

        h, w = arr.shape[-2], arr.shape[-1]
        target_h, target_w = target_hw
        if h < target_h or w < target_w:
            return arr.reshape(-1)

        trimmed_h = (h // target_h) * target_h
        trimmed_w = (w // target_w) * target_w
        arr = arr[:trimmed_h, :trimmed_w]
        block_h = trimmed_h // target_h
        block_w = trimmed_w // target_w
        arr = arr.reshape(target_h, block_h, target_w, block_w)
        arr = arr.mean(axis=(1, 3))
        return arr.reshape(-1)


class MoEAgent(nn.Module):
    """Policy/value network that learns to fuse expert policies (plus optional pixels)."""

    def __init__(self, envs, device, layer_dims=(128, 64), weighted_sum=True):
        """Build the MoE controller with configurable hidden layers and fusion mode."""
        super().__init__()
        self.device = device

        base_env = envs
        while hasattr(base_env, "venv"):
            base_env = base_env.venv
        self.action_dim = base_env.get_attr("action_dim")[0]
        self.num_experts = base_env.get_attr("num_experts")[0]
        self.policy_feature_dim = base_env.get_attr("policy_feature_dim")[0]
        self.raw_obs_dim = base_env.get_attr("raw_obs_dim")[0]
        self.encoded_raw_hw = base_env.get_attr("encoded_raw_hw")[0]

        # Branch that reasons over expert policy outputs
        policy_hidden_dim = layer_dims[0] if len(layer_dims) > 0 else 128
        self.policy_branch = nn.Sequential(
            layer_init(nn.Linear(self.policy_feature_dim, policy_hidden_dim)),
            nn.ReLU(),
        )

        # Optional branch encoding the low-resolution pixel grid
        if self.raw_obs_dim > 0 and self.encoded_raw_hw is not None:
            raw_h, raw_w = self.encoded_raw_hw
            self.raw_branch = nn.Sequential(
                layer_init(
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                layer_init(
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            conv_out_h = raw_h // 4
            conv_out_w = raw_w // 4
            conv_out_dim = 16 * conv_out_h * conv_out_w
            raw_hidden_dim = policy_hidden_dim
            self.raw_project = nn.Sequential(
                layer_init(nn.Linear(conv_out_dim, raw_hidden_dim)),
                nn.ReLU(),
            )
        else:
            self.raw_branch = None
            self.raw_project = None
            raw_hidden_dim = 0

        # Fuse both branches (if present) and optionally stack more MLP layers
        fusion_input_dim = policy_hidden_dim + raw_hidden_dim
        fusion_layers: list[nn.Module] = []
        current_dim = fusion_input_dim
        for hidden_dim in layer_dims[1:]:
            fusion_layers.append(layer_init(
                nn.Linear(current_dim, hidden_dim)))
            fusion_layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.fusion_network = nn.Sequential(
            *fusion_layers) if fusion_layers else nn.Identity()
        feature_dim = current_dim if fusion_layers else fusion_input_dim

        if weighted_sum:
            output_dim = self.num_experts
            self.weighted_sum = True
        else:
            output_dim = self.action_dim
            self.weighted_sum = False

        self.actor = layer_init(nn.Linear(feature_dim, output_dim), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1)

    def _forward_features(self, x):
        policy_part = x[:, :self.policy_feature_dim]
        policy_features = self.policy_branch(policy_part)

        if self.raw_branch is not None and self.raw_project is not None:
            raw_part = x[:, self.policy_feature_dim:]
            raw_h, raw_w = self.encoded_raw_hw
            raw_img = raw_part.view(x.size(0), 1, raw_h, raw_w)
            raw_features = self.raw_branch(raw_img)
            raw_features = self.raw_project(raw_features)
            features = torch.cat((policy_features, raw_features), dim=1)
        else:
            features = policy_features

        return self.fusion_network(features)

    def get_value(self, x):
        """Estimate the value function for PPO updates."""
        hidden = self._forward_features(x)
        return self.critic(hidden)

    def _direct_distribution(self, hidden):
        """Return a categorical distribution directly from the actor head."""
        logits = self.actor(hidden)
        return Categorical(logits=logits)

    def _weighted_distribution(self, hidden, x):
        """Blend per-expert policies using the learned mixture weights."""
        weights = torch.softmax(self.actor(hidden), dim=1)
        policy_part = x[:, :self.policy_feature_dim]
        experts = policy_part.reshape(
            x.size(0), self.num_experts, self.action_dim)
        mixed_probs = torch.sum(weights.unsqueeze(-1) * experts, dim=1)
        mixed_probs = torch.clamp(mixed_probs, min=1e-8)
        mixed_probs = mixed_probs / mixed_probs.sum(dim=1, keepdim=True)
        return Categorical(probs=mixed_probs)

    def get_action_and_value(self, x, action=None):
        """Sample/score actions and obtain entropy+value for PPO optimisation."""
        hidden = self._forward_features(x)
        if self.weighted_sum:
            categorical = self._weighted_distribution(hidden, x)
        else:
            categorical = self._direct_distribution(hidden)
        if action is None:
            action = categorical.sample()
        return action, categorical.log_prob(action), categorical.entropy(), self.critic(hidden)


# Global variable to hold parsed arguments
global args


# Function to create a gym environment with the specified settings
def make_env(env_id, idx, capture_video, run_dir, target_device):
    """Factory that builds a fully wrapped Atari environment for each worker."""
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
            env = gym.make(env_id, render_mode="rgb_array",
                           frameskip=args.frameskip)
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

        wrapper_device = torch.device(target_device)
        if args.include_raw_obs:
            env = RawObservationCache(env)
        env = ocatari_wrappers.MultiOCCAMWrapper(env, wrappers=args.moe_wrappers)
        # Replace observation with concatenated expert policies (and optional pixels)
        env = MoEWrapper(
            env,
            args.model_dir,
            wrapper_device,
            args.include_raw_obs,
            args.raw_obs_downsample,
        )

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
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create RTPT object to monitor progress with estimated time remaining
    rtpt = RTPT(name_initials=args.author, experiment_name=f'PPObj-v2_{args.env_id.split("ALE/")[-1].split("-v")[0]}',
                max_iterations=args.num_iterations)

    # Set logger level and determine whether to use GPU or CPU for computation
    logger.set_level(args.logging_level)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # expose for subprocesses that import this module
    globals()["device"] = device
    logger.debug(f"Using device {device}.")

    # Environment setup
    envs = SubprocVecEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, writer_dir, str(device)
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

    agent = MoEAgent(envs, device, args.layer_dims).to(device)

    # Initialize optimizer for training
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Allocate storage for observations, actions, rewards, etc.
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.action_space.shape).to(device)
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

    # Start the RTPT tracking
    rtpt.start()

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
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and store reward, next observation, and done flag
            next_obs, reward, next_done, infos = envs.step(
                action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(
                device), torch.Tensor(next_done).to(device)

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
                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approximate KL divergence
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # Normalize advantages
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss (for exploration)
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backpropagation and optimizer step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Compute explained variance (diagnostic measure for value function fit quality)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # Log episode statistics for Tensorboard
        if done_in_episode:
            if args.new_rf:
                writer.add_scalar("charts/Episodic_New_Reward",
                                  enewr / count, global_step)
            writer.add_scalar("charts/Episodic_Original_Reward",
                              eorgr / count, global_step)
            writer.add_scalar("charts/Episodic_Length",
                              elength / count, global_step)
            pbar.set_description(f"Reward: {eorgr.item() / count:.1f}")

        # Log other statistics
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)

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
