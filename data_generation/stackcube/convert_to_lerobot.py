"""Convert replayed ManiSkill StackCube-v1 demos (rgb obs, pd_ee_delta_pos) into a LeRobot v3.0 dataset.

Written through LeRobot's own `LeRobotDataset.create/add_frame/save_episode/finalize`, so the
result loads with the official `lerobot` library (the generic mani_skill/trajectory/convert_to_lerobot.py
writes `data_files_size_in_mb: 0`, which lerobot>=0.6 rejects, and encodes lossy mp4v).

Input: the output of
    python -m mani_skill.trajectory.replay_trajectory --traj-path .../StackCube-v1/motionplanning/trajectory.h5 \
        --use-first-env-state -c pd_ee_delta_pos -o rgb --save-traj --num-envs 10 -b physx_cpu --count 120
which keeps only replays that end in success. The first `--num-episodes` trajectories, in replay
order, become episodes 0..N-1.

Run (lerobot is not a dependency of this repo):
    uv run --no-project --with "lerobot[dataset]==0.6.1" --with h5py \
        python data_generation/stackcube/convert_to_lerobot.py --h5 <replayed.h5> --output-dir <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

log = logging.getLogger("stackcube_to_lerobot")

FPS = 20  # StackCube-v1 control_freq
TASK = "Pick up the red cube and stack it on the green cube."
CAMERAS = ("base_camera", "hand_camera")  # panda_wristcam: fixed scene camera + wrist camera, 128x128
FEATURES = {
    **{
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channels"],
        }
        for cam in CAMERAS
    },
    # Joint positions: 7 arm joints [rad], then the 2 finger joints [m].
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),
        "names": [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"],
    },
    # pd_ee_delta_pos, normalized to [-1, 1]: xyz change of the TCP target position in the robot
    # base frame (1.0 = 0.1 m), and the gripper target (-1 = closed, +1 = open).
    "action": {
        "dtype": "float32",
        "shape": (4,),
        "names": ["tcp_delta_x", "tcp_delta_y", "tcp_delta_z", "gripper"],
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, required=True, help="Replayed trajectory .h5 (its .json must sit next to it).")
    p.add_argument("--output-dir", type=Path, required=True, help="Must not exist yet.")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--repo-id", default="recast-robotics/maniskill-stackcube-100")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    meta = json.loads(args.h5.with_suffix(".json").read_text())
    env_kwargs = meta["env_info"]["env_kwargs"]
    if (meta["env_info"]["env_id"], env_kwargs["control_mode"], env_kwargs["obs_mode"]) != (
        "StackCube-v1",
        "pd_ee_delta_pos",
        "rgb",
    ):
        raise ValueError(f"Expected a StackCube-v1 / pd_ee_delta_pos / rgb replay, got {meta['env_info']}.")

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES,
        root=args.output_dir,
        robot_type="panda_wristcam",
        use_videos=True,
    )
    with h5py.File(args.h5, "r") as f:
        # Numeric order: h5py iterates keys as strings (traj_0, traj_1, traj_10, ...).
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        if len(keys) < args.num_episodes:
            raise ValueError(f"{args.h5} has {len(keys)} trajectories, fewer than --num-episodes={args.num_episodes}.")
        episodes = {e["episode_id"]: e for e in meta["episodes"]}
        for episode_index, key in enumerate(keys[: args.num_episodes]):
            t = f[key]
            actions = np.asarray(t["actions"], dtype=np.float32)  # (N, 4)
            qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float32)  # (N+1, 9)
            rgb = {cam: np.asarray(t[f"obs/sensor_data/{cam}/rgb"]) for cam in CAMERAS}  # (N+1, 128, 128, 3)
            n = len(actions)
            if not bool(t["success"][-1]):
                raise ValueError(f"{key} does not end in success; the dataset must hold successful demos only.")
            if qpos.shape[0] != n + 1 or any(v.shape[0] != n + 1 for v in rgb.values()):
                raise ValueError(f"{key}: expected {n + 1} observations for {n} actions.")
            # Frame i pairs observation i with the action taken from it; the terminal observation is dropped.
            for i in range(n):
                ds.add_frame(
                    {
                        **{f"observation.images.{cam}": rgb[cam][i] for cam in CAMERAS},
                        "observation.state": qpos[i],
                        "action": actions[i],
                        "task": TASK,
                    }
                )
            ds.save_episode()
            source = episodes[int(key.split("_")[1])]
            log.info(
                "episode %d <- %s (ManiSkill episode_seed %d): %d frames",
                episode_index, key, source["episode_seed"], n,
            )
    ds.finalize()
    log.info("Wrote %d episodes, %d frames to %s", ds.meta.total_episodes, ds.meta.total_frames, args.output_dir)


if __name__ == "__main__":
    main()
