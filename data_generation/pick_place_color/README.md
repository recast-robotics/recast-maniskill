# PickAndPlaceColor-v1 data generation

Two stages: collect motion-planning demonstrations (states + actions only), then render
them into a LeRobot dataset with both cameras.

## 0. mplib / numpy workaround

`mplib==0.1.1` is built against numpy 1.x and **segfaults** at `Planner.__init__` under
numpy >= 2 (both environments on this machine ship numpy 2.x). Shim a numpy 1.x into the
path for stage 1 only — stage 2 does not use mplib and runs fine on numpy 2:

```bash
python -m pip install --target=/tmp/np1 "numpy<2"
export PYTHONPATH=/home/kelin/recast-maniskill:/tmp/np1
```

## 1. Motion planning demos

```bash
python -m mani_skill.examples.motionplanning.panda.run \
    -e PickAndPlaceColor-v1 -n 600 --only-count-success -b cpu \
    --record-dir /home/kelin/dataset/maniskill/pick_place_color --traj-name trajectory
```

~1.3 s per episode, ~99% of seeds solve. Keep `--num-procs 1`: with multiple processes
each worker walks forward from its own start seed on failure, so the per-process seed
ranges overlap and a few episodes come out duplicated.

Output: `PickAndPlaceColor-v1/motionplanning/trajectory.h5` (env states + actions,
`obs_mode=none`, ~220 MB for 600 episodes) and the matching `.json`.

## 2. LeRobot dataset

```bash
python data_generation/pick_place_color/build_lerobot.py
```

Replays every recorded frame by environment state and re-renders `base_camera` and
`hand_camera` at 480x480 (see the module docstring). ~5 s per episode, ~23 MB per episode
video-encoded. Storing the same RGB as raw arrays in an h5 instead would cost ~240 MB per
episode, which is why the source h5 stays observation-free and the images live in video.

### Rendering in parallel

600 episodes take ~45 min in one process. Sharding cuts that to ~14 min:

```bash
for i in 0 1 2 3; do
  python data_generation/pick_place_color/build_lerobot.py \
      --num-shards 4 --shard $i --shard-mode block \
      --repo-id "maniskill/pick_place_color_s$i" \
      --out "$SHARDS_TMP/s$i" --overwrite &
done
wait
```

then merge the shard roots **in episode order** with lerobot's `aggregate_datasets`, which
keeps `episode_index` equal to the index of the trajectory in the source h5:

```python
from lerobot.datasets.aggregate import aggregate_datasets
aggregate_datasets(
    repo_ids=[f"maniskill/pick_place_color_s{i}" for i in range(4)],
    roots=[SHARDS_TMP / f"s{i}" for i in range(4)],
    aggr_repo_id="maniskill/pick_place_color",
    aggr_root=LEROBOT_ROOT,
)
```

**Cap the shard count at 4 on a 30 GB machine.** Each renderer holds ~5 GB RSS, so 6
shards get one OOM-killed by the kernel part-way through. Always require every shard to
print its `DONE` line before merging -- a merge over a silently truncated shard produces a
dataset that looks valid but is missing episodes. Verify after merging:

```python
ds = LeRobotDataset("recast-robotics/pick-color-cube-600")
assert ds.num_frames == sum(e["elapsed_steps"] for e in json.load(open(H5_JSON))["episodes"])
```

`--shard-mode stride` (round-robin) is also available, but interleaves episodes across
shards, so the merged `episode_index` no longer matches the source trajectory index.

## 3. Upload to the Hub

```bash
python data_generation/pick_place_color/upload_to_hub.py          # private
python data_generation/pick_place_color/upload_to_hub.py --public
```

Pushes to `recast-robotics/pick-color-cube-600` with videos, a LeRobot dataset card and
the `v3.0` codebase tag. Uses `upload_large_folder=True` so a ~14 GB push resumes rather
than restarting after a network drop.


## Locations

| What | Where |
|---|---|
| Motion-planning source | `/home/kelin/dataset/maniskill/pick_place_color/PickAndPlaceColor-v1/motionplanning/trajectory.h5` |
| LeRobot dataset | `~/.cache/huggingface/lerobot/recast-robotics/pick-color-cube-600` |
| Hub (private) | `recast-robotics/pick-color-cube-600` |

The LeRobot dataset lives under `HF_LEROBOT_HOME`, so it loads with no `root=` argument
and no download:

```python
LeRobotDataset("recast-robotics/pick-color-cube-600")
```
