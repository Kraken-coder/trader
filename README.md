---
title: Trader Environment Server
emoji: 🎤
colorFrom: yellow
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Trader Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Trader environment is through the `TraderEnv` class:

```python
from trader import TraderAction, TraderEnv

try:
    # Create environment from Docker image
    traderenv = TraderEnv.from_docker_image("trader-env:latest")

    # Reset
    result = traderenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = traderenv.step(TraderAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    traderenv.close()
```

That's it! The `TraderEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t trader-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**TraderAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**TraderObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Trader environment server running, you can connect directly:

```python
from trader import TraderEnv

# Connect to existing server
traderenv = TraderEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = traderenv.reset()
result = traderenv.step(TraderAction(message="Hello!"))
```

Note: When connecting to an existing server, `traderenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from trader import TraderAction, TraderEnv

# Connect with context manager (auto-connects and closes)
with TraderEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(TraderAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TraderEnvironment,  # Pass class, not instance
    TraderAction,
    TraderObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from trader import TraderAction, TraderEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with TraderEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(TraderAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

## Trading Schema

The current trader environment uses a position-based action schema:

```python
TraderAction(
    position="long",  # long | short | noop
    take_profit_price=0.0,
    stop_loss_price=0.0,
)
```

Observations include both engineered indicators and the raw candle fields:
`open_price`, `high_price`, `low_price`, `current_price`, along with position and
account state such as `position_sign`, `entry_price`, `drawdown`, and `pnl`.

## Qwen SFT Seed

Build a JSONL seed corpus from the parquet data:

```bash
python scripts/build_qwen_sft_corpus.py --output sft_corpus/qwen_trader_seed.jsonl
```

Then fine-tune Qwen with LoRA on that corpus:

```bash
python scripts/train_qwen_sft.py \
    --dataset sft_corpus/qwen_trader_seed.jsonl \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --output-dir qwen_trader_sft \
    --epochs 1 \
    --train-batch-size 2 \
    --grad-accum 8
```

After SFT, continue with GRPO-style optimization on the live environment reward oracle:

```bash
python scripts/train_qwen_grpo.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path qwen_trader_sft \
    --output-dir qwen_trader_grpo \
    --steps 200 \
    --group-size 4 \
    --max-holding-bars 60
```

The environment's holding window is now explicit and defaults to `60` bars via `max_holding_bars`.

## Requirements For Qwen SFT

For an RTX 4060, install a CUDA-enabled PyTorch build first, then install the SFT extras:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install -e ".[sft]"
```

Minimum runtime requirements:
- Python 3.10+
- NVIDIA driver compatible with the PyTorch CUDA wheel you install
- CUDA-enabled PyTorch
- `datasets`, `accelerate`, `transformers`, `peft`

Recommended runtime flags on RTX 4060:
- Leave `--bf16` off unless your local PyTorch build reports bf16 support
- Keep LoRA enabled for memory savings
- Use the default `fp16` path if bf16 is not supported

The trainer requires CUDA and will fail fast if no NVIDIA GPU is available.

## Train an MLP RL Agent

You can train a baseline MLP actor-critic agent directly against this environment:

```bash
python train_mlp_agent.py --episodes 1500 --episode-length 1024 --log-every 25
```

For 1-minute candles, the trainer now exposes reward-shaping knobs:

```bash
python train_mlp_agent.py \
    --episodes 2000 \
    --episode-length 1024 \
    --fee-rate 0.0002 \
    --reward-scale 200 \
    --idle-penalty 0.10 \
    --flip-penalty 0.00005 \
    --reward-clip 0.03 \
    --entropy-coef 0.01 \
    --eval-episodes 5
```

Training outputs are written to `training_outputs/`:

- `training_metrics.csv`: per-episode train/eval rewards and moving average
- `mlp_actor_critic.pt`: saved model checkpoint
- `training_curve.png`: reward curve (if `matplotlib` is installed)

To verify that learning improves over time, compare:

- early episode average reward (first ~50 episodes)
- late episode average reward (last ~50 episodes)

The script prints this summary automatically at the end.

## Train a Rainbow-Style DQN Agent (Recommended)

For 1-minute multi-coin data, use off-policy training for better sample efficiency:

```bash
python train_rainbow_dqn.py \
    --episodes 2500 \
    --episode-length 1024 \
    --frame-stack 16 \
    --batch-size 512 \
    --replay-size 500000 \
    --warmup-steps 20000 \
    --n-step 5 \
    --eval-every 25 \
    --eval-episodes 8
```

Outputs are written to `training_outputs_rainbow/`:

- `rainbow_metrics.csv`
- `rainbow_dqn.pt`
- `rainbow_curve.png`

This trainer includes Double DQN, Dueling heads, Prioritized Replay, N-step returns, and Distributional C51 targets.

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/trader_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
trader/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TraderEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── trader_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
