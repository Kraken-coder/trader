"""Train an MLP-based RL agent on TraderEnvironment.

This script uses a simple on-policy actor-critic objective for the discrete
action space: long, short, noop, close.

Outputs:
- training_metrics.csv
- mlp_actor_critic.pt
- training_curve.png (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from models import MarketFeatures, TraderAction
from server.trader_environment import TraderEnvironment


ACTION_SPACE = ["long", "short", "noop"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_SPACE)}
INDEX_TO_ACTION = {idx: name for idx, name in enumerate(ACTION_SPACE)}
OHLC_FEATURE_COLUMNS = ("open_price", "high_price", "low_price", "close_price")


@dataclass
class EpisodeStats:
    episode: int
    train_reward: float
    eval_reward: float
    moving_avg_50: float


class MLPActorCritic(nn.Module):
    """Simple MLP policy/value network for tabular feature vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, len(ACTION_SPACE))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value


def obs_to_tensor(observation) -> torch.Tensor:
    features = observation.market_features.model_dump()
    vec = [float(features[name]) for name in MarketFeatures.FEATURE_COLUMNS]
    vec.extend(float(getattr(observation, name)) for name in OHLC_FEATURE_COLUMNS)
    return torch.tensor(vec, dtype=torch.float32)


def discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def run_episode(
    env: TraderEnvironment,
    model: MLPActorCritic,
    gamma: float,
    entropy_coef: float,
    value_coef: float,
    train: bool,
) -> Tuple[float, torch.Tensor | None]:
    obs = env.reset()

    log_probs: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    rewards: List[float] = []
    entropies: List[torch.Tensor] = []

    done = bool(obs.done)
    episode_reward = 0.0

    while not done:
        state = obs_to_tensor(obs).unsqueeze(0)
        logits, value = model(state)
        dist = torch.distributions.Categorical(logits=logits)

        if train:
            action_idx = int(dist.sample().item())
        else:
            action_idx = int(torch.argmax(logits, dim=-1).item())

        action = TraderAction(
            position=INDEX_TO_ACTION[action_idx],
            take_profit_price=0.0,
            stop_loss_price=0.0,
        )
        next_obs = env.step(action)

        reward = float(next_obs.reward or 0.0)
        done = bool(next_obs.done)

        episode_reward += reward

        if train:
            log_probs.append(dist.log_prob(torch.tensor(action_idx)))
            values.append(value.squeeze(0))
            rewards.append(reward)
            entropies.append(dist.entropy().squeeze(0))

        obs = next_obs

    if not train:
        return episode_reward, None

    returns = discounted_returns(rewards, gamma)
    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    entropy_t = torch.stack(entropies)

    advantages = returns - values_t.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = value_coef * (returns - values_t).pow(2).mean()
    entropy_bonus = entropy_t.mean()

    total_loss = policy_loss + value_loss - entropy_coef * entropy_bonus
    return episode_reward, total_loss


def maybe_plot_curve(csv_path: Path, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    episodes = []
    train_rewards = []
    eval_rewards = []
    moving_avg = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            train_rewards.append(float(row["train_reward"]))
            eval_rewards.append(float(row["eval_reward"]))
            moving_avg.append(float(row["moving_avg_50"]))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, train_rewards, alpha=0.35, label="Train reward")
    plt.plot(episodes, moving_avg, linewidth=2.0, label="Train moving avg (50)")
    plt.plot(episodes, eval_rewards, linewidth=1.5, label="Eval reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MLP RL Agent Learning Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def train(args: argparse.Namespace) -> None:
    train_env = TraderEnvironment(
        episode_length=args.episode_length,
        slippage_bps=args.slippage_bps,
        drawdown_penalty=args.drawdown_penalty,
        default_take_profit_pct=args.take_profit_pct,
        default_stop_loss_pct=args.stop_loss_pct,
        seed=args.seed,
    )
    eval_env = TraderEnvironment(
        episode_length=args.episode_length,
        slippage_bps=args.slippage_bps,
        drawdown_penalty=args.drawdown_penalty,
        default_take_profit_pct=args.take_profit_pct,
        default_stop_loss_pct=args.stop_loss_pct,
        seed=args.seed + 999,
    )

    input_dim = len(MarketFeatures.FEATURE_COLUMNS) + len(OHLC_FEATURE_COLUMNS)
    model = MLPActorCritic(input_dim=input_dim, hidden_dim=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "training_metrics.csv"
    model_path = out_dir / "mlp_actor_critic.pt"
    curve_path = out_dir / "training_curve.png"

    reward_buffer: List[float] = []
    stats_rows: List[EpisodeStats] = []

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "train_reward", "eval_reward", "moving_avg_50"])

        for ep in range(1, args.episodes + 1):
            model.train()
            train_reward, loss = run_episode(
                train_env,
                model,
                args.gamma,
                args.entropy_coef,
                args.value_coef,
                train=True,
            )
            assert loss is not None

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            reward_buffer.append(train_reward)
            window = reward_buffer[-50:]
            moving = sum(window) / len(window)

            model.eval()
            with torch.no_grad():
                eval_scores = []
                for _ in range(args.eval_episodes):
                    score, _ = run_episode(
                        eval_env,
                        model,
                        args.gamma,
                        args.entropy_coef,
                        args.value_coef,
                        train=False,
                    )
                    eval_scores.append(score)
                eval_reward = sum(eval_scores) / len(eval_scores)

            row = EpisodeStats(
                episode=ep,
                train_reward=train_reward,
                eval_reward=eval_reward,
                moving_avg_50=moving,
            )
            stats_rows.append(row)
            writer.writerow([row.episode, row.train_reward, row.eval_reward, row.moving_avg_50])

            if ep % args.log_every == 0 or ep == 1:
                print(
                    f"Episode {ep:5d} | train={train_reward: .6f} | "
                    f"eval={eval_reward: .6f} | mov50={moving: .6f}"
                )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "action_space": ACTION_SPACE,
            "feature_columns": list(MarketFeatures.FEATURE_COLUMNS),
            "ohlc_feature_columns": list(OHLC_FEATURE_COLUMNS),
            "episodes": args.episodes,
        },
        model_path,
    )

    maybe_plot_curve(metrics_path, curve_path)

    first_window = stats_rows[: min(50, len(stats_rows))]
    last_window = stats_rows[max(0, len(stats_rows) - 50) :]
    first_avg = sum(x.train_reward for x in first_window) / max(1, len(first_window))
    last_avg = sum(x.train_reward for x in last_window) / max(1, len(last_window))

    print("\nTraining complete.")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    if curve_path.exists():
        print(f"Saved curve: {curve_path}")
    else:
        print("Curve not saved (matplotlib not installed).")
    print(
        f"Learning summary | first_window_avg={first_avg:.6f} | "
        f"last_window_avg={last_avg:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP RL agent on TraderEnvironment")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--episode-length", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.10)
    parser.add_argument("--take-profit-pct", type=float, default=0.012)
    parser.add_argument("--stop-loss-pct", type=float, default=0.006)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", type=str, default="training_outputs")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
