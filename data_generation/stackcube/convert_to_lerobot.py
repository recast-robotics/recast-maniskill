"""Convert replayed ManiSkill StackCube-v1 demos (rgb obs, pd_ee_delta_pos) into a LeRobot v3.0 dataset.

Written through LeRobot's own `LeRobotDataset.create/add_frame/save_episode/finalize`, so the
result loads with the official `lerobot` library (the generic mani_skill/trajectory/convert_to_lerobot.py
writes `data_files_size_in_mb: 0`, which lerobot>=0.6 rejects, and encodes lossy mp4v).

Input: one or more trajectory .h5 files in ManiSkill's layout, each with its .json, e.g. the output of
    python -m mani_skill.trajectory.replay_trajectory --traj-path .../StackCube-v1/motionplanning/trajectory.h5 \
        --use-first-env-state -c pd_ee_delta_pos -o rgb --save-traj --num-envs 10 -b physx_cpu --count 120
(which keeps only replays that end in success), or act-minimal's `collect_rollouts.py` output. The first
`--num-episodes` trajectories of each file, in order, are appended file after file.

Run (lerobot is not a dependency of this repo):
    uv run --no-project --with "lerobot[dataset]==0.6.1" --with h5py \
        python data_generation/stackcube/convert_to_lerobot.py --h5 <a.h5> [<b.h5> ...] --output-dir <dir>
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

# Optional (--with-outcomes): the outcome of the episode each frame belongs to, constant per episode.
OUTCOME_FEATURE = {"episode_success": {"dtype": "bool", "shape": (1,), "names": None}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--h5",
        type=Path,
        nargs="+",
        required=True,
        help="Trajectory .h5 files (each with its .json next to it), appended in the order given.",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Must not exist yet.")
    p.add_argument(
        "--num-episodes",
        type=int,
        nargs="+",
        default=[100],
        help="Episodes taken from the start of each --h5 file (one value per file, or one for all).",
    )
    p.add_argument("--repo-id", default="recast-robotics/maniskill-stackcube-100")
    p.add_argument(
        "--with-outcomes",
        action="store_true",
        help="Accept failed episodes and add a per-frame bool `episode_success` feature (the episode's "
        "outcome, i.e. whether its last step is a success). Without it every episode must succeed.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    counts = args.num_episodes * len(args.h5) if len(args.num_episodes) == 1 else args.num_episodes
    if len(counts) != len(args.h5):
        raise ValueError(f"Got {len(args.h5)} --h5 files but {len(args.num_episodes)} --num-episodes values.")

    metas = []
    for h5 in args.h5:
        meta = json.loads(h5.with_suffix(".json").read_text())
        env_kwargs = meta["env_info"]["env_kwargs"]
        if (meta["env_info"]["env_id"], env_kwargs["control_mode"], env_kwargs["obs_mode"]) != (
            "StackCube-v1",
            "pd_ee_delta_pos",
            "rgb",
        ):
            raise ValueError(f"{h5}: expected StackCube-v1 / pd_ee_delta_pos / rgb, got {meta['env_info']}.")
        metas.append(meta)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES | (OUTCOME_FEATURE if args.with_outcomes else {}),
        root=args.output_dir,
        robot_type="panda_wristcam",
        use_videos=True,
    )
    episode_index = 0
    for h5, meta, count in zip(args.h5, metas, counts, strict=True):
        first = episode_index
        with h5py.File(h5, "r") as f:
            # Numeric order: h5py iterates keys as strings (traj_0, traj_1, traj_10, ...).
            keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
            if len(keys) < count:
                raise ValueError(f"{h5} has {len(keys)} trajectories, fewer than the {count} requested.")
            episodes = {e["episode_id"]: e for e in meta["episodes"]}
            for key in keys[:count]:
                t = f[key]
                actions = np.asarray(t["actions"], dtype=np.float32)  # (N, 4)
                qpos = np.asarray(t["obs/agent/qpos"], dtype=np.float32)  # (N+1, 9)
                rgb = {cam: np.asarray(t[f"obs/sensor_data/{cam}/rgb"]) for cam in CAMERAS}  # (N+1, 128, 128, 3)
                n = len(actions)
                success = bool(t["success"][-1])
                if not success and not args.with_outcomes:
                    raise ValueError(
                        f"{h5}:{key} does not end in success; pass --with-outcomes to include failed episodes."
                    )
                if qpos.shape[0] != n + 1 or any(v.shape[0] != n + 1 for v in rgb.values()):
                    raise ValueError(f"{h5}:{key}: expected {n + 1} observations for {n} actions.")
                # Frame i pairs observation i with the action taken from it; the terminal observation is dropped.
                for i in range(n):
                    ds.add_frame(
                        {
                            **{f"observation.images.{cam}": rgb[cam][i] for cam in CAMERAS},
                            "observation.state": qpos[i],
                            "action": actions[i],
                            "task": TASK,
                            **({"episode_success": np.array([success])} if args.with_outcomes else {}),
                        }
                    )
                ds.save_episode()
                source = episodes[int(key.split("_")[1])]
                log.info(
                    "episode %d <- %s:%s (seed %d): %d frames",
                    episode_index, h5.name, key, source["episode_seed"], n,
                )
                episode_index += 1
        print(f"episodes {first}-{episode_index - 1} <- {h5} (first {count} trajectories)", flush=True)
    ds.finalize()
    print(f"Wrote {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
