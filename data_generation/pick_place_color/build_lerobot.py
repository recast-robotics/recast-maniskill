"""Render ManiSkill PickAndPlaceColor-v1 motion-planning demos into a LeRobot v3.0 dataset.

The source .h5 is collected with ``obs_mode=none`` (states + actions only, ~200 MB for
600 episodes). This script replays it by *environment state*: every recorded frame is
written back into the simulator with ``set_state_dict`` and both cameras are re-rendered.
No physics is re-simulated, so the trajectory is bit-identical to the source; only the
observations are produced here. That keeps the source .h5 small and lets the dataset be
re-rendered at any resolution / camera setup without re-running motion planning.

Recorded per frame:
  observation.images.base_camera  (H, W, 3) uint8, video-encoded
  observation.images.hand_camera  (H, W, 3) uint8, video-encoded  (panda_wristcam)
  observation.state               qpos (9,)
  observation.qvel                qvel (9,)
  action                          pd_joint_pos target (8,)

Sharding: pass --num-shards N and --shard i to render a disjoint slice of the episodes
into its own dataset root; the shards can then be combined with lerobot's
``aggregate_datasets``. A single process renders ~5 s per episode, so sharding is only
worth it for much larger collections.
"""

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401  (registers PickAndPlaceColor-v1)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ENV_ID = "PickAndPlaceColor-v1"
FPS = 20  # PickAndPlaceColor-v1 control_freq
TASK = "Pick up each cube and place it into the tray of the same color."
CAMS = ["base_camera", "hand_camera"]

DEFAULT_H5 = (
    "/home/kelin/dataset/maniskill/pick_place_color/PickAndPlaceColor-v1/"
    "motionplanning/trajectory.h5"
)
DEFAULT_OUT = "/home/kelin/.cache/huggingface/lerobot/recast-robotics/pick-color-cube-600"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default=DEFAULT_H5)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--repo-id", default="maniskill/pick_place_color")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--count", type=int, default=None, help="only render the first N episodes")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument(
        "--shard-mode",
        choices=["block", "stride"],
        default="block",
        help="block: contiguous episode ranges, so aggregating the shards in order "
        "preserves episode_index == traj index. stride: round-robin.",
    )
    p.add_argument("--image-writer-processes", type=int, default=0)
    p.add_argument("--image-writer-threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_features(height: int, width: int) -> dict:
    features = {
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
        for cam in CAMS
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (9,),
                "names": [f"qpos_{i}" for i in range(9)],
            },
            "observation.qvel": {
                "dtype": "float32",
                "shape": (9,),
                "names": [f"qvel_{i}" for i in range(9)],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"action_{i}" for i in range(8)],
            },
        }
    )
    return features


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=build_features(args.height, args.width),
        root=out,
        robot_type="panda",
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    env = gym.make(
        ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        num_envs=1,
        sensor_configs=dict(width=args.width, height=args.height),
    )
    u = env.unwrapped
    device = u.device

    f = h5py.File(args.h5, "r")
    traj_keys = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
    if args.count is not None:
        traj_keys = traj_keys[: args.count]
    if args.shard_mode == "stride":
        traj_keys = traj_keys[args.shard :: args.num_shards]
    else:
        # contiguous block; the remainder is spread over the first shards
        n_total = len(traj_keys)
        base, rem = divmod(n_total, args.num_shards)
        start = args.shard * base + min(args.shard, rem)
        stop = start + base + (1 if args.shard < rem else 0)
        traj_keys = traj_keys[start:stop]

    # The scene only has to be built once; every episode is fully described by the
    # stored env_states, so a single reset up front is enough.
    env.reset(seed=0)

    total = 0
    for n_ep, k in enumerate(traj_keys):
        t = f[k]
        actions = np.asarray(t["actions"], dtype=np.float32)  # (N, 8)
        actor_states = {
            name: np.asarray(t[f"env_states/actors/{name}"])
            for name in t["env_states/actors"]
        }
        art_states = {
            name: np.asarray(t[f"env_states/articulations/{name}"])
            for name in t["env_states/articulations"]
        }
        n = actions.shape[0]

        for i in range(n):  # obs[i] pairs with action[i]; the terminal obs is dropped
            state = {
                "actors": {
                    name: torch.as_tensor(arr[i], dtype=torch.float32, device=device)[None]
                    for name, arr in actor_states.items()
                },
                "articulations": {
                    name: torch.as_tensor(arr[i], dtype=torch.float32, device=device)[None]
                    for name, arr in art_states.items()
                },
            }
            u.set_state_dict(state)
            obs = u.get_obs()

            frame = {
                f"observation.images.{cam}": obs["sensor_data"][cam]["rgb"][0]
                .cpu()
                .numpy()
                .astype(np.uint8)
                for cam in CAMS
            }
            frame["observation.state"] = (
                u.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
            )
            frame["observation.qvel"] = (
                u.agent.robot.get_qvel()[0].cpu().numpy().astype(np.float32)
            )
            frame["action"] = actions[i]
            frame["task"] = TASK
            ds.add_frame(frame)

        ds.save_episode()
        total += n
        print(f"[shard {args.shard}] {k}: {n} frames ({n_ep + 1}/{len(traj_keys)})", flush=True)

    f.close()
    env.close()
    print(f"DONE shard {args.shard}: {len(traj_keys)} episodes, {total} frames -> {out}")


if __name__ == "__main__":
    main()
