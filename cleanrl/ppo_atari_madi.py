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

from typing import Literal

from architectures.madi import MaskerNet

# -----------------------
# Warnings & determinism
# -----------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# cuBLAS deterministic workspace (required by torch.use_deterministic_algorithms on CUDA)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# -----------------------
# Optional OC_ATARI_DIR
# -----------------------
oc_atari_dir = os.getenv("OC_ATARI_DIR")
if oc_atari_dir is not None:
    oc_atari_path = os.path.join(Path(__file__), oc_atari_dir)
    sys.path.insert(1, oc_atari_path)

# -----------------------
# Evals import
# -----------------------
eval_dir = os.path.join(Path(__file__).parent.parent, "cleanrl_utils/evals/")
sys.path.insert(1, eval_dir)
from generic_eval import evaluate  # noqa


# -----------------------
# Args
# -----------------------
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
    obs_mode: Literal["dqn"] = "dqn"
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

    # Tracking
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "OCCAM"
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
    author: str = "JD"
    """Initials of the author"""
    checkpoint_interval: int = 40
    """Number of iterations before a model checkpoint is saved and uploaded to wandb"""

    # Algorithm-specific arguments
    architecture: str = "PPO"
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

    # Transformer parameters
    emb_dim: int = 128
    """input embedding size of the transformer"""
    num_heads: int = 64
    """number of multi-attention heads"""
    num_blocks: int = 1
    """number of transformer blocks"""
    patch_size: int = 12
    """ViT patch size"""

    # MaDi parameters
    masker_lr: float = 1e-3
    """learning rate for the masking network"""
    masker_beta: float = 0.9
    """beta for the masking network adam optimizer"""
    masker_num_layers: int = 2
    """number of layers for the MaDi masker"""
    masker_num_filters: int = 32
    """number of filters for the MaDi masker"""


    # PPObj network parameters
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

    # runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    masked_wrapper: str = None
    add_pixels: bool = False


# Global for access inside thunks
global args


# -----------------------
# Helpers
# -----------------------
def seed_everything(seed: int, cuda: bool = True, torch_deterministic: bool = True):
    """Seed python, numpy, torch (+cuda) and set deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(torch_deterministic)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = False


def _log_model_artifact(run, path, name, iteration=None, metadata=None):
    import wandb
    aliases = ["latest"]
    if iteration is not None:
        aliases.append(f"iter-{iteration}")
    art = wandb.Artifact(name=name, type="model", metadata=metadata or {})
    art.add_file(path)
    run.log_artifact(art, aliases=aliases)


# -----------------------
# Env factory with per-worker seeding
# -----------------------
def make_env(env_id, idx, seed, capture_video, run_dir):
    """
    Creates a gym environment with the specified settings and seeds it.
    """
    def thunk():
        # Per-subprocess RNG seeds (VERY important)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        logger.set_level(args.logging_level)

        # Backend selection
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
            env = gym.make(env_id, render_mode="rgb_array", frameskip=args.frameskip)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, args.buffer_window_size)
        else:
            raise ValueError("Unknown Backend")

        # Capture video from env #0
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"{run_dir}/media/videos", disable_logger=True)

        # Standard Atari wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        # Seed env + spaces via Gymnasium API
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)

        return env

    return thunk


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Parse args & seed everything globally
    args = tyro.cli(Args)

    # Runtime-dependent
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Prepare masking wrapper flags
    if "masked" in args.obs_mode:
        if args.obs_mode.endswith("+pixels"):
            args.masked_wrapper = args.obs_mode[:-7]
            args.add_pixels = True
        else:
            args.masked_wrapper = args.obs_mode
            args.add_pixels = False
        args.obs_mode = "ori"

    # Global seeding
    seed_everything(args.seed, cuda=args.cuda, torch_deterministic=args.torch_deterministic)

    # Run name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # W&B init (optional)
    if args.track:
        import dataclasses, wandb

        wb_dir = args.wandb_dir or str(Path("runs") / run_name)
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=dataclasses.asdict(args),
            sync_tensorboard=True,
            save_code=True,
            dir=wb_dir,
            job_type="train",
            group=f"{args.env_id}_{args.architecture}",
            tags=[args.env_id, args.architecture, args.backend, args.obs_mode, "MaDi"],
            resume="allow",
        )
        wandb.define_metric("global_step")
        wandb.define_metric("charts/*", step_metric="global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
        wandb.define_metric("time/*", step_metric="global_step")
        writer_dir = run.dir
        postfix = dict(url=run.url)
    else:
        writer_dir = str(Path(args.wandb_dir or ".") / "runs" / run_name)
        postfix = None

    # TB writer
    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # RTPT
    rtpt = RTPT(name_initials=args.author, experiment_name='OCALM', max_iterations=args.num_iterations)
    rtpt.start()

    # Device
    logger.set_level(args.logging_level)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.debug(f"Using device {device}.")

    # Vectorized envs with per-worker seeds
    envs = SubprocVecEnv(
        [make_env(args.env_id, i, args.seed + i, args.capture_video, writer_dir) for i in range(args.num_envs)]
    )
    envs = VecNormalize(envs, norm_obs=False, norm_reward=True)

    # Agent
    if args.architecture == "PPO":
        from architectures.ppo import PPODefault as Agent
        agent = Agent(envs, device, normalize=False).to(device)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} does not exist!")

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    masker = MaskerNet(obs_shape=envs.observation_space.shape, num_layers=args.masker_num_layers,
                       num_filters=args.masker_num_filters).to(device)
    masker_optimizer = torch.optim.Adam(masker.parameters(), lr=args.masker_lr, betas=(args.masker_beta, 0.999))

    if args.track:
        num_params = sum(p.numel() for p in agent.parameters())
        wandb.summary["params_total"] = num_params
        wandb.watch(agent, log="gradients", log_freq=1000, log_graph=False)

    # Allocate rollout storage (clear dtypes, on device)
    obs_space_shape = envs.observation_space.shape
    act_space_shape = envs.action_space.shape
    obs      = torch.zeros((args.num_steps, args.num_envs) + obs_space_shape, dtype=torch.float32, device=device)
    actions  = torch.zeros((args.num_steps, args.num_envs) + act_space_shape, dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards  = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones    = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values   = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    pbar = tqdm(range(1, args.num_iterations + 1), postfix=postfix)
    for iteration in pbar:
        # LR anneal
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Episode collectors / aggregates
        episode_returns, episode_lengths = [], []
        elength = 0
        eorgr = 0.0
        enewr = 0.0
        count = 0

        # Checkpoint
        if iteration % args.checkpoint_interval == 0:
            model_path = f"{writer_dir}/{args.exp_name}_{iteration}.cleanrl_model"
            model_data = {
                "model_weights": agent.state_dict(),
                "args": vars(args),
                "Timesteps": iteration * args.batch_size
            }
            torch.save(model_data, model_path)
            logger.info(f"model saved to {model_path} at iteration {iteration}")
            if args.track:
                _log_model_artifact(
                    run, model_path, name=f"{args.exp_name}",
                    iteration=iteration, metadata={"env": args.env_id, "seed": args.seed}
                )

        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                masked_obs = masker(next_obs)
                action, logprob, _, value = agent.get_action_and_value(masked_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, next_done_np, infos = envs.step(action.detach().cpu().numpy())
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            # Per-episode stats
            if bool(next_done.bool().any()):
                for info in infos:
                    if "episode" in info:
                        r = float(info["episode"]["r"])
                        l = int(info["episode"]["l"])
                        episode_returns.append(r)
                        episode_lengths.append(l)
                        count += 1
                        elength += l
                        if args.new_rf:
                            enewr += r
                            eorgr += float(info.get("org_return", 0.0))
                        else:
                            eorgr += r

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + obs_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + act_space_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy & value
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                masked_obs = masker(b_obs[mb_inds])
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    masked_obs, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                masker_optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(masker.parameters(), args.max_grad_norm)
                optimizer.step()
                masker_optimizer.step()

                if args.track and (start % (args.minibatch_size * 4) == 0):
                    import wandb
                    wandb.log({"losses/grad_total_norm": float(gn)}, step=global_step)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Explained variance
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TensorBoard scalars (remain)
        if count > 0:
            if args.new_rf:
                writer.add_scalar("charts/Episodic_New_Reward", enewr / count, global_step)
            writer.add_scalar("charts/Episodic_Original_Reward", eorgr / count, global_step)
            writer.add_scalar("charts/Episodic_Length", elength / count, global_step)
            pbar.set_description(f"Reward: {eorgr / count:.1f}")

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # W&B enrichments
        if args.track:
            import wandb
            log_payload = {
                "global_step": global_step,
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": float(np.mean(clipfracs)),
                "losses/explained_variance": float(explained_var),
                "time/SPS": int(global_step / (time.time() - start_time)),
            }
            if count > 0:
                if args.new_rf:
                    log_payload["charts/Episodic_New_Reward"] = enewr / count
                log_payload["charts/Episodic_Original_Reward"] = eorgr / count
                log_payload["charts/Episodic_Length"] = elength / count
            if episode_returns:
                log_payload["charts/ReturnHist"] = wandb.Histogram(episode_returns)
                log_payload["charts/LengthHist"] = wandb.Histogram(episode_lengths)
            if device.type == "cuda":
                log_payload["sys/gpu_mem_alloc_GB"] = torch.cuda.memory_allocated() / 1e9
                log_payload["sys/gpu_mem_reserved_GB"] = torch.cuda.memory_reserved() / 1e9
            wandb.log(log_payload, step=global_step)

        rtpt.step()

    # Final save
    model_path = f"{writer_dir}/{args.exp_name}_final.cleanrl_model"
    model_data = {"model_weights": agent.state_dict(), "args": vars(args)}
    torch.save(model_data, model_path)
    logger.info(f"model saved to {model_path} at final")

    masker_path = f"{writer_dir}/{args.exp_name}_final.cleanrl_model"
    masker_data = {"model_weights": masker.state_dict(), "args": vars(args)}
    torch.save(masker_data, masker_path)
    logger.info(f"masker saved to {masker_path} at final")

    if args.track:
        import wandb
        _log_model_artifact(run, model_path, name=f"{args.exp_name}",
                            iteration=None, metadata={"final": True})

        _log_model_artifact(run, masker_path, name=f"{args.exp_name}_masker",
                            iteration=None, metadata={"final": True})

        # Evaluate agent's performance
        class EvalAgent(nn.Module):
            def __init__(self):
                super().__init__()
                self.masker = masker
                self.agent = agent

            def get_action_and_value(self, x):
                return self.agent.get_action_and_value(self.masker(x))

        args.new_rf = ""
        rewards = evaluate(
            EvalAgent(), make_env, 10,
            env_id=args.env_id, capture_video=args.capture_video,
            run_dir=writer_dir, device=device, seed=args.seed
        )
        wandb.summary["FinalReward_mean"] = float(np.mean(rewards))
        wandb.summary["FinalReward_median"] = float(np.median(rewards))
        wandb.summary["FinalReward_min"] = float(np.min(rewards))
        wandb.summary["FinalReward_max"] = float(np.max(rewards))
        wandb.log({"eval/RewardHist": wandb.Histogram(rewards)}, step=global_step)

        # if args.test_modifs != "":
        #     args.modifs = args.test_modifs
        #     args.backend = "HackAtari"
        #     rewards = evaluate(
        #         agent, make_env, 10,
        #         env_id=args.env_id, capture_video=args.capture_video,
        #         run_dir=writer_dir, device=device
        #     )
        #     wandb.log({"HackAtariReward": np.mean(rewards)}, step=global_step)

        # Optional: videos to W&B + artifact (visible in UI)
        if args.capture_video:
            import os, glob
            video_dir = f"{writer_dir}/media/videos"
            videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
            if videos:
                # Log a couple of the most recent videos to the Media panel
                for v in videos[-2:]:
                    wandb.log({"video": wandb.Video(v, fps=30, format="mp4")}, step=global_step)
                # Force-upload all videos to the Files tab immediately
                wandb.save(os.path.join(video_dir, "*.mp4"), policy="now")

                # Also keep a versioned artifact with all videos
                va = wandb.Artifact(f"{args.exp_name}-videos", type="videos")
                for v in videos:
                    va.add_file(v)
                run.log_artifact(va, aliases=["latest"])

        wandb.finish()

    envs.close()
    writer.close()
