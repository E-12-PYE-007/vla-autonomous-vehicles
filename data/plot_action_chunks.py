import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

data_dir = Path(__file__).parent
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(data_dir.glob("action_chunks_*.csv"))[-1]
seq_num = int(sys.argv[2]) if len(sys.argv) > 2 else 75
df = pd.read_csv(csv_path)
df = df[df["seq_num"] == seq_num]

# image_path columns in the CSV are relative to wherever the ROS node was launched
# from, not to the CSV file. The images dir is always a sibling of the CSV
# (data/images_<run_id>/), so resolve against that instead of trusting the stored
# path's base.
run_id = csv_path.stem.removeprefix("action_chunks_")
images_dir = csv_path.parent / f"images_{run_id}"


def resolve_image(column: str, source: str | None = None) -> tuple[Path, str] | None:
    if column not in df.columns:
        return None
    rows = df[df["source"] == source] if source is not None else df
    paths = rows[column].dropna()
    paths = paths[paths != ""]
    if paths.empty:
        return None
    name = Path(paths.iloc[0]).name
    resolved = images_dir / name
    if not resolved.exists():
        return None
    img_seq = name.removeprefix("seq_").split(".")[0]
    return resolved, img_seq


# "image_path" is the frame at seq_num, shared by sys1 and sys2 (sys1's hidden
# state input and sys2's only input). "curr_image_path" is sys1's extra, newer
# frame (sys1 runs faster than sys2, so a fresher /cam frame is usually available
# by the time it publishes) -- for sys2 it's identical to image_path.
past_image = resolve_image("image_path")
curr_image = resolve_image("curr_image_path", source="sys1")
if curr_image == past_image:
    curr_image = None

images = []
if past_image is not None:
    path, img_seq = past_image
    images.append((path, f"Frame at img_seq={img_seq} (sys1 hidden state input, sys2's only input)"))
if curr_image is not None:
    path, img_seq = curr_image
    images.append((path, f"sys1's newer frame, img_seq={img_seq}"))

n_img = len(images)
if n_img == 0:
    fig, ax = plt.subplots(figsize=(8, 8))
else:
    fig, axes = plt.subplots(1, n_img + 1, figsize=(8 * (n_img + 1), 8))
    img_axes, ax = axes[:n_img], axes[n_img]
    for ax_img, (path, title) in zip(img_axes, images):
        ax_img.imshow(mpimg.imread(path))
        ax_img.set_title(title, fontsize=10)
        ax_img.axis("off")

colors = {"sys1": "tab:blue", "sys2": "tab:orange"}
markers = {"sys1": "o", "sys2": "s"}
for source, sub in df.groupby("source"):
    sub = sub.sort_values("waypoint_idx")
    ax.plot(sub["x"], sub["y"], color=colors[source], marker=markers[source],
            markersize=8, linewidth=1.5, label=source)
    for _, row in sub.iterrows():
        ax.annotate(int(row["waypoint_idx"]), (row["x"], row["y"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color=colors[source])

ax.set_xlabel("x, relative to robot pose at chunk time (m)")
ax.set_ylabel("y, relative to robot pose at chunk time (m)")
ax.set_title(f"Waypoints for seq {seq_num}: {csv_path.name}")
ax.set_aspect("equal", adjustable="datalim")
ax.legend()

goal_text = None
if "goal" in df.columns:
    goals = df["goal"].dropna()
    goals = goals[goals != ""]
    if not goals.empty:
        goal_text = goals.iloc[0]

if goal_text:
    fig.suptitle(f'Goal: "{goal_text}"', fontsize=13)

fig.tight_layout()

out_path = data_dir / (csv_path.stem + f"_seq{seq_num}_plot.png")
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
