"""Record an MP4 video for cooperative simple_tag.

This script runs a short rollout with random chaser actions (or a trained model if you load one)
and records an MP4 under ./videos.

Why a standalone script?
- Keeps training code untouched.
- Uses PettingZoo render_mode='rgb_array' to capture frames.

Usage (PowerShell):
    python ./MADDPG/record_video_simple_tag.py --episodes 1 --max-cycles 200

Note:
- Requires pettingzoo + imageio + imageio-ffmpeg (already in repo requirements.txt).
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import imageio

# Ensure repo root is on sys.path so `MADDPG.*` imports work when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def make_env(max_cycles: int = 200, coop_tag: bool = True):
    from importlib import import_module

    scenario = "simple_tag_v3"
    mod = import_module(f"pettingzoo.mpe.{scenario}")

    raw_env = mod.parallel_env(
        max_cycles=max_cycles,
        continuous_actions=True,
        render_mode="rgb_array",
    )

    if coop_tag:
        from MADDPG.common.simple_tag_coop import SimpleTagCoopParallelEnv

        raw_env = SimpleTagCoopParallelEnv(raw_env)

    return raw_env


def sample_action(space):
    # PettingZoo MPE continuous action is Box(5,) in [0,1]
    a = np.random.uniform(0.0, 1.0, size=space.shape).astype(np.float32)
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--no-coop", action="store_true", help="do not wrap coop_tag")
    args = parser.parse_args()

    env = make_env(max_cycles=args.max_cycles, coop_tag=(not args.no_coop))

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(args.out_dir, f"simple_tag_coop_{stamp}.mp4")

    writer = imageio.get_writer(out_path, fps=args.fps)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()

            # record first frame
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

            for t in range(args.max_cycles):
                actions = {}
                for agent in env.agents:
                    space = env.action_space(agent)
                    actions[agent] = sample_action(space)

                obs, rew, term, trunc, info = env.step(actions)

                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)

                if any(term.values()) or any(trunc.values()):
                    break

            # small pause between episodes so video doesn't look too abrupt
            time.sleep(0.05)

    finally:
        writer.close()
        env.close()

    print(f"Saved video -> {out_path}")


if __name__ == "__main__":
    main()
