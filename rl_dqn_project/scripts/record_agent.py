"""Render GIF evidence of an actual saved trained policy."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
import gymnasium as gym

from agents.tabular import TabularQLearning
from agents.dqn import DQNAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a trained agent acting greedily.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", choices=("CartPole-v1", "Acrobot-v1"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.suffix == ".npz":
        agent = TabularQLearning.load(str(checkpoint))
    elif checkpoint.suffix == ".pt":
        agent = DQNAgent.load(checkpoint)
    else:
        raise ValueError("Expected a .npz or .pt checkpoint.")
    env = gym.make(args.env_id, render_mode="rgb_array")
    frames = []
    try:
        observation, _ = env.reset(seed=args.seed)
        frames.append(env.render())
        while True:
            action = agent.greedy_act(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
    finally:
        env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=args.fps, loop=0)
    print(f"Saved {len(frames)} real environment frames to {output}")


if __name__ == "__main__":
    main()
