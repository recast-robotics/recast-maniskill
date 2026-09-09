"""Upload the PickAndPlaceColor-v1 LeRobot dataset to the Hugging Face Hub.

Requires an HF token with write access to the target org (`hf auth login`, or HF_TOKEN).
Defaults to a PRIVATE repo; pass --public to publish publicly.

Note on the local metadata check below: LeRobotDatasetMetadata falls back to downloading
`meta/` from the Hub when the local files are missing, which 404s for a local-only
repo_id and produces a confusing error. The explicit check fails fast instead, and makes
it obvious when the local dataset is not where it is expected to be.
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# HF_LEROBOT_HOME layout, so LeRobotDataset(DST_REPO_ID) finds it with no root= and
# no download.
ROOT = Path("/home/kelin/.cache/huggingface/lerobot/recast-robotics/pick-color-cube-5000-mixed")
# Only a label: LeRobotDataset reads meta/info.json from --root and never consults
# the Hub while it is present, so this just has to be stable, not resolvable.
SRC_REPO_ID = "recast-robotics/pick-color-cube-5000-mixed"
DST_REPO_ID = "recast-robotics/pick-color-cube-5000-mixed"

parser = argparse.ArgumentParser()
parser.add_argument("--public", action="store_true", help="create a public repo instead of private")
parser.add_argument("--repo-id", default=DST_REPO_ID)
parser.add_argument("--root", default=ROOT, type=Path)
args = parser.parse_args()

private = not args.public

info_path = args.root / "meta" / "info.json"
if not info_path.is_file():
    raise SystemExit(
        f"No LeRobot dataset at {args.root} (missing {info_path}).\n"
        "Build it first with data_generation/pick_and_place_color/build_lerobot.py."
    )

ds = LeRobotDataset(SRC_REPO_ID, root=args.root, video_backend="pyav")

# push_to_hub() uses self.repo_id for create_repo/upload_folder/card/tag.
ds.repo_id = args.repo_id
ds.meta.repo_id = args.repo_id

print(f"Uploading {ds.num_episodes} episodes / {ds.num_frames} frames")
print(f"  -> {args.repo_id} (private={private})")

ds.push_to_hub(
    private=private,
    tags=["maniskill", "robotics", "panda", "simulation", "pick-and-place", "color-sorting"],
    push_videos=True,
    # resumable + parallel; the dataset is ~14 GB of video
    upload_large_folder=True,
)

print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")
