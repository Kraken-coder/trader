"""Train a Rainbow-style DQN agent on TraderEnvironment.

Implemented components:
- Double DQN
- Dueling network head
- Prioritized replay (proportional)
- N-step returns
- Distributional RL (C51)
- Optional frame stacking for short-term temporal context

Outputs:
- rainbow_metrics.csv
- rainbow_dqn.pt
- rainbow_curve.png (if matplotlib installed)
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MarketFeatures, TraderAction
from server.trader_environment import TraderEnvironment


ACTION_SPACE = ["long", "short", "noop"]
INDEX_TO_ACTION = {idx: name for idx, name in enumerate(ACTION_SPACE)}
OHLC_FEATURE_COLUMNS = ("open_price", "high_price", "low_price", "close_price")


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """Simple proportional prioritized replay buffer."""

    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition, priority: float | None = None) -> None:
        if priority is None:
            max_prio = float(self.priorities.max()) if self.buffer else 1.0
            priority = max_prio

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = float(priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        probs = np.power(prios + 1e-8, self.alpha)
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = np.power(len(self.buffer) * probs[indices], -beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(max(prio, 1e-6))


class NStepAccumulator:
    """Build n-step transitions online before inserting into replay."""

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: Deque[Transition] = deque(maxlen=n_step)

    def _make_n_step(self) -> Transition:
        first = self.buffer[0]
        reward = 0.0
        next_state = self.buffer[-1].next_state
        done = self.buffer[-1].done

        for i, tr in enumerate(self.buffer):
            reward += (self.gamma**i) * tr.reward
            next_state = tr.next_state
            if tr.done:
                done = True
                break

        return Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def push(self, transition: Transition) -> Transition | None:
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return None
        out = self._make_n_step()
        self.buffer.popleft()
        return out

    def flush(self) -> List[Transition]:
        out: List[Transition] = []
        while self.buffer:
            out.append(self._make_n_step())
            self.buffer.popleft()
        return out


class DuelingC51(nn.Module):
    """Dueling C51 network for discrete actions."""

    def __init__(self, input_dim: int, action_dim: int, num_atoms: int, hidden_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_atoms),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * num_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        v = self.value(z).unsqueeze(1)
        a = self.advantage(z).view(-1, self.action_dim, self.num_atoms)
        q_atoms = v + (a - a.mean(dim=1, keepdim=True))
        return q_atoms


class RainbowAgent:
    def __init__(self, args: argparse.Namespace, input_dim: int, device: torch.device):
        self.args = args
        self.device = device

        self.num_atoms = args.num_atoms
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=device)

        self.online = DuelingC51(
            input_dim=input_dim,
            action_dim=len(ACTION_SPACE),
            num_atoms=self.num_atoms,
            hidden_dim=args.hidden_dim,
        ).to(device)
        self.target = DuelingC51(
            input_dim=input_dim,
            action_dim=len(ACTION_SPACE),
            num_atoms=self.num_atoms,
            hidden_dim=args.hidden_dim,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=args.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def q_values(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return torch.sum(probs * self.support.view(1, 1, -1), dim=-1)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(len(ACTION_SPACE))

        with torch.no_grad():
            state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            logits = self.online(state_t)
            q = self.q_values(logits)
            return int(torch.argmax(q, dim=1).item())

    def project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma_n: float,
    ) -> torch.Tensor:
        batch_size = rewards.size(0)
        projected = torch.zeros(batch_size, self.num_atoms, device=self.device)

        tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * self.support.unsqueeze(0)
        tz = tz.clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / self.delta_z
        l = torch.floor(b).long()
        u = torch.ceil(b).long()

        # Guard against floating-point edge cases at vmax that can produce
        # an out-of-range atom index on CUDA (num_atoms instead of num_atoms-1).
        l = l.clamp(0, self.num_atoms - 1)
        u = u.clamp(0, self.num_atoms - 1)

        offset = (
            torch.arange(batch_size, device=self.device).unsqueeze(1) * self.num_atoms
        )

        projected_flat = projected.view(-1)

        eq_mask = l == u
        projected_flat.index_add_(
            0,
            (l + offset).view(-1),
            (next_dist * (u.float() - b + eq_mask.float())).view(-1),
        )
        projected_flat.index_add_(
            0,
            (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1),
        )

        return projected

    def train_step(
        self,
        replay: PrioritizedReplayBuffer,
        beta: float,
        gamma_n: float,
    ) -> float:
        batch, indices, weights_np = replay.sample(self.args.batch_size, beta)

        states = torch.from_numpy(np.stack([t.state for t in batch])).float().to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).float().to(self.device)
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)
        weights = torch.from_numpy(weights_np).float().to(self.device)

        with torch.no_grad():
            online_next_logits = self.online(next_states)
            next_actions = torch.argmax(self.q_values(online_next_logits), dim=1)

            target_next_logits = self.target(next_states)
            next_action_logits = target_next_logits[
                torch.arange(target_next_logits.size(0), device=self.device),
                next_actions,
            ]
            next_dist = torch.softmax(next_action_logits, dim=1)

            target_dist = self.project_distribution(next_dist, rewards, dones, gamma_n)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.online(states)
            chosen_logits = logits[
                torch.arange(logits.size(0), device=self.device),
                actions,
            ]
            log_probs = F.log_softmax(chosen_logits, dim=1)

            per_sample_loss = -(target_dist * log_probs).sum(dim=1)
            loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online.parameters(), self.args.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        replay.update_priorities(indices, per_sample_loss.detach().cpu().numpy() + 1e-6)
        return float(loss.detach().cpu().item())

    def soft_update_target(self, tau: float) -> None:
        with torch.no_grad():
            for t_param, o_param in zip(self.target.parameters(), self.online.parameters()):
                t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)


class FrameStack:
    def __init__(self, state_dim: int, stack_size: int):
        self.state_dim = state_dim
        self.stack_size = stack_size
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, first_state: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(first_state.copy())
        return self._get()

    def step(self, next_state: np.ndarray) -> np.ndarray:
        self.frames.append(next_state.copy())
        return self._get()

    def _get(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)


def obs_to_vector(obs) -> np.ndarray:
    feats = obs.market_features.model_dump()
    vec = [float(feats[c]) for c in MarketFeatures.FEATURE_COLUMNS]
    vec.extend(float(getattr(obs, name)) for name in OHLC_FEATURE_COLUMNS)
    return np.array(vec, dtype=np.float32)


def epsilon_by_step(step: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    if step >= eps_decay_steps:
        return eps_end
    ratio = step / float(eps_decay_steps)
    return eps_start + ratio * (eps_end - eps_start)


def evaluate(
    env: TraderEnvironment,
    agent: RainbowAgent,
    episodes: int,
    frame_stack: int,
) -> float:
    rewards = []
    state_dim = len(MarketFeatures.FEATURE_COLUMNS)

    for _ in range(episodes):
        obs = env.reset()
        base_state = obs_to_vector(obs)
        fs = FrameStack(state_dim, frame_stack)
        state = fs.reset(base_state)

        done = bool(obs.done)
        ep_reward = 0.0

        while not done:
            action_idx = agent.act(state, epsilon=0.0)
            next_obs = env.step(
                TraderAction(
                    position=INDEX_TO_ACTION[action_idx],
                    take_profit_price=0.0,
                    stop_loss_price=0.0,
                )
            )
            next_state = fs.step(obs_to_vector(next_obs))

            ep_reward += float(next_obs.reward or 0.0)
            done = bool(next_obs.done)
            state = next_state

        rewards.append(ep_reward)

    return float(np.mean(rewards)) if rewards else 0.0


def maybe_plot(csv_path: Path, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    episodes = []
    train_rewards = []
    eval_rewards = []
    moving_avg = []
    losses = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            train_rewards.append(float(row["train_reward"]))
            eval_rewards.append(float(row["eval_reward"]))
            moving_avg.append(float(row["moving_avg_50"]))
            losses.append(float(row["avg_loss"]))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(episodes, train_rewards, alpha=0.3, label="Train reward")
    axes[0].plot(episodes, moving_avg, linewidth=2.0, label="Train moving avg (50)")
    axes[0].plot(episodes, eval_rewards, linewidth=1.5, label="Eval reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Rainbow DQN Learning Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(episodes, losses, color="tab:red", alpha=0.9, label="Avg train loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Rainbow-style DQN on TraderEnvironment")

    p.add_argument("--episodes", type=int, default=2500)
    p.add_argument("--episode-length", type=int, default=1024)
    p.add_argument("--frame-stack", type=int, default=16)

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=5)

    p.add_argument("--num-atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=-0.15)
    p.add_argument("--v-max", type=float, default=0.15)

    p.add_argument("--replay-size", type=int, default=500000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=20000)
    p.add_argument("--updates-per-step", type=int, default=1)

    p.add_argument("--prio-alpha", type=float, default=0.6)
    p.add_argument("--prio-beta-start", type=float, default=0.4)
    p.add_argument("--prio-beta-end", type=float, default=1.0)

    p.add_argument("--eps-start", type=float, default=0.20)
    p.add_argument("--eps-end", type=float, default=0.02)
    p.add_argument("--eps-decay-steps", type=int, default=300000)

    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--max-grad-norm", type=float, default=10.0)

    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=8)

    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--drawdown-penalty", type=float, default=0.10)
    p.add_argument("--take-profit-pct", type=float, default=0.012)
    p.add_argument("--stop-loss-pct", type=float, default=0.006)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="training_outputs_rainbow")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_train = TraderEnvironment(
        episode_length=args.episode_length,
        slippage_bps=args.slippage_bps,
        drawdown_penalty=args.drawdown_penalty,
        default_take_profit_pct=args.take_profit_pct,
        default_stop_loss_pct=args.stop_loss_pct,
        seed=args.seed,
    )
    env_eval = TraderEnvironment(
        episode_length=args.episode_length,
        slippage_bps=args.slippage_bps,
        drawdown_penalty=args.drawdown_penalty,
        default_take_profit_pct=args.take_profit_pct,
        default_stop_loss_pct=args.stop_loss_pct,
        seed=args.seed + 101,
    )

    base_dim = len(MarketFeatures.FEATURE_COLUMNS) + len(OHLC_FEATURE_COLUMNS)
    input_dim = base_dim * args.frame_stack

    agent = RainbowAgent(args, input_dim=input_dim, device=device)
    replay = PrioritizedReplayBuffer(capacity=args.replay_size, alpha=args.prio_alpha)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rainbow_metrics.csv"
    ckpt_path = out_dir / "rainbow_dqn.pt"
    curve_path = out_dir / "rainbow_curve.png"

    total_env_steps = 0
    moving_rewards: List[float] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "train_reward", "eval_reward", "moving_avg_50", "avg_loss", "buffer_size", "epsilon"])

        for ep in range(1, args.episodes + 1):
            obs = env_train.reset()
            fs = FrameStack(base_dim, args.frame_stack)
            state = fs.reset(obs_to_vector(obs))

            done = bool(obs.done)
            ep_reward = 0.0
            losses: List[float] = []
            nstep = NStepAccumulator(n_step=args.n_step, gamma=args.gamma)

            while not done:
                eps = epsilon_by_step(total_env_steps, args.eps_start, args.eps_end, args.eps_decay_steps)
                action_idx = agent.act(state, epsilon=eps)

                next_obs = env_train.step(
                    TraderAction(
                        position=INDEX_TO_ACTION[action_idx],
                        take_profit_price=0.0,
                        stop_loss_price=0.0,
                    )
                )
                next_state = fs.step(obs_to_vector(next_obs))
                reward = float(next_obs.reward or 0.0)
                done = bool(next_obs.done)

                one_step = Transition(
                    state=state,
                    action=action_idx,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                n_tr = nstep.push(one_step)
                if n_tr is not None:
                    replay.add(n_tr)

                if done:
                    for tr in nstep.flush():
                        replay.add(tr)

                state = next_state
                ep_reward += reward
                total_env_steps += 1

                if len(replay) >= max(args.warmup_steps, args.batch_size):
                    beta = args.prio_beta_start + (
                        min(1.0, total_env_steps / max(1, args.eps_decay_steps))
                        * (args.prio_beta_end - args.prio_beta_start)
                    )
                    gamma_n = args.gamma ** args.n_step
                    for _ in range(args.updates_per_step):
                        loss = agent.train_step(replay, beta=beta, gamma_n=gamma_n)
                        losses.append(loss)
                        agent.soft_update_target(args.target_tau)

            moving_rewards.append(ep_reward)
            mov50 = float(np.mean(moving_rewards[-50:]))

            if ep % args.eval_every == 0 or ep == 1:
                eval_reward = evaluate(env_eval, agent, args.eval_episodes, args.frame_stack)
            else:
                eval_reward = 0.0

            avg_loss = float(np.mean(losses)) if losses else 0.0
            writer.writerow([
                ep,
                ep_reward,
                eval_reward,
                mov50,
                avg_loss,
                len(replay),
                eps,
            ])

            if ep % args.log_every == 0 or ep == 1:
                print(
                    f"Episode {ep:5d} | train={ep_reward: .6f} | eval={eval_reward: .6f} | "
                    f"mov50={mov50: .6f} | loss={avg_loss: .6f} | "
                    f"buf={len(replay)} | eps={eps: .4f}"
                )

    torch.save(
        {
            "online_state_dict": agent.online.state_dict(),
            "target_state_dict": agent.target.state_dict(),
            "input_dim": input_dim,
            "base_feature_dim": base_dim,
            "frame_stack": args.frame_stack,
            "feature_columns": list(MarketFeatures.FEATURE_COLUMNS),
            "ohlc_feature_columns": list(OHLC_FEATURE_COLUMNS),
            "action_space": ACTION_SPACE,
            "config": vars(args),
        },
        ckpt_path,
    )

    maybe_plot(csv_path, curve_path)

    first_avg = float(np.mean(moving_rewards[: min(50, len(moving_rewards))]))
    last_avg = float(np.mean(moving_rewards[max(0, len(moving_rewards) - 50) :]))

    print("\nTraining complete.")
    print(f"Saved model: {ckpt_path}")
    print(f"Saved metrics: {csv_path}")
    if curve_path.exists():
        print(f"Saved curve: {curve_path}")
    else:
        print("Curve not saved (matplotlib not installed).")
    print(
        f"Learning summary | first_window_avg={first_avg:.6f} | "
        f"last_window_avg={last_avg:.6f}"
    )


if __name__ == "__main__":
    main()
