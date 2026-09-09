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

### Starting part-way through the task

Episodes can begin with some cubes already delivered, which is useful for weighting the
dataset toward the later stages. The stage numbers refer to the 6-stage reward:

| `start_stage` | starts with | cubes to move | reward at reset | mean episode |
|---|---|---|---|---|
| 1 | nothing placed (the full task) | 3 | 0 | ~305 frames |
| 3 | red delivered | 2 | 2 | ~208 frames |
| 5 | red and green delivered | 1 | 4 | ~109 frames |

Set the mix with `start_stage_probs`, as weights over stages 1 / 3 / 5:

```bash
python -m mani_skill.examples.motionplanning.panda.run \
    -e PickAndPlaceColor-v1 -n 600 --only-count-success -b cpu \
    --record-dir /home/kelin/dataset/maniskill/pick_place_color --traj-name trajectory \
    --env-kwargs '{"start_stage_probs": [0.5, 0.25, 0.25]}'
```

The stage is drawn per episode from the episode RNG, so a given seed always produces the
same stage and the whole mix is reproducible. `reset(options={"start_stage": k})` pins it
for a single episode and overrides the probabilities. Over 90 episodes the example above
realised 51.1% / 25.6% / 23.3%.

The chosen mix is recorded in the trajectory `.json` under `env_info.env_kwargs`, and an
individual episode's starting stage can be recovered from its first frame by counting how
many cubes already sit in their trays.

Output: `PickAndPlaceColor-v1/motionplanning/trajectory.h5` (env states + actions,
`obs_mode=none`, ~220 MB for 600 episodes) and the matching `.json`.

## 2. LeRobot dataset

```bash
python data_generation/pick_and_place_color/build_lerobot.py
```

Replays every recorded frame by environment state and re-renders `base_camera` and
`hand_camera` at 480x480 (see the module docstring). ~5 s per episode, ~23 MB per episode
video-encoded. Storing the same RGB as raw arrays in an h5 instead would cost ~240 MB per
episode, which is why the source h5 stays observation-free and the images live in video.

### Rendering in parallel

600 episodes take ~45 min in one process. Sharding cuts that to ~14 min:

```bash
for i in 0 1 2 3; do
  python data_generation/pick_and_place_color/build_lerobot.py \
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

**Cap the shard count at 4 on a 30 GB machine.** Each renderer holds ~3.5-5 GB RSS, so 6
shards get one OOM-killed by the kernel part-way through. Always require every shard to
print its `DONE` line before merging -- a merge over a silently truncated shard produces a
dataset that looks valid but is missing episodes. Verify after merging:

```python
ds = LeRobotDataset("recast-robotics/pick-color-cube-5000-mixed")
assert ds.num_frames == sum(e["elapsed_steps"] for e in json.load(open(H5_JSON))["episodes"])
```

**Keep `--image-writer-threads 0`** (the default). LeRobot's async image writer queues
frames in memory and, when it outruns the video encoder, the backlog grows without bound.
Rendering 2000 episodes, one of three otherwise identical shards reached 8.8 GB RSS after
138 episodes while its siblings sat at 4.4 GB, and an earlier shard was OOM-killed at
9.3 GB. Because the growth does not track episode count, smaller shards do not avoid it.
Writing synchronously removes the queue and holds memory flat at ~3.5 GB, with no
measurable throughput cost.

`--shard-mode stride` (round-robin) is also available, but interleaves episodes across
shards, so the merged `episode_index` no longer matches the source trajectory index.

## 3. Upload to the Hub

```bash
python data_generation/pick_and_place_color/upload_to_hub.py          # private
python data_generation/pick_and_place_color/upload_to_hub.py --public
```

Pushes to `recast-robotics/pick-color-cube-5000-mixed` with videos, a LeRobot dataset card and
the `v3.0` codebase tag. Uses `upload_large_folder=True` so a ~14 GB push resumes rather
than restarting after a network drop.


## Locations

| What | Where |
|---|---|
| Motion-planning source | `/home/kelin/dataset/maniskill/pick_place_color/PickAndPlaceColor-v1/motionplanning/trajectory.h5` |
| LeRobot dataset | `~/.cache/huggingface/lerobot/recast-robotics/pick-color-cube-5000-mixed` |
| Hub (private) | `recast-robotics/pick-color-cube-5000-mixed` |

The LeRobot dataset lives under `HF_LEROBOT_HOME`, so it loads with no `root=` argument
and no download:

```python
LeRobotDataset("recast-robotics/pick-color-cube-5000-mixed")
```
