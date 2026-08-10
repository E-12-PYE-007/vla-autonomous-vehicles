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


# The three frames spanning one sys2 inference:
#   image_path      -- what went into sys2, and the frame sys1 looks up for its hidden
#                      state input. Shared by both.
#   end_image_path  -- newest frame to arrive by the time sys2's forward pass returned.
#   curr_image_path -- what sys1 actually paired with at its own tick (it runs faster
#                      than sys2, so this is usually fresher still). Equals image_path
#                      for sys2, which only conditions on one frame.
# Any of them can coincide; duplicates are dropped so a panel is only shown per distinct
# frame.
past_image = resolve_image("image_path")
end_image = resolve_image("end_image_path", source="sys1")
curr_image = resolve_image("curr_image_path", source="sys1")

labelled = [
    (past_image, "Into sys2 / sys1 hidden state input"),
    (end_image, "Newest when sys2 finished"),
    (curr_image, "What sys1 paired with"),
]

images = []
seen = set()
for resolved, label in labelled:
    if resolved is None or resolved in seen:
        continue
    seen.add(resolved)
    path, img_seq = resolved
    images.append((path, f"{label}\nimg_seq={img_seq}"))

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
    # The action head only outputs the 8 future waypoints, not the robot's own
    # current pose, so waypoint_idx 0 is already offset from (0, 0). Prepend the
    # origin so the line starts where the robot actually is (matches upstream
    # run_omnivla.py / run_asyncvla.py, which np.insert a (0, 0) start point).
    xs = [0.0] + sub["x"].tolist()
    ys = [0.0] + sub["y"].tolist()
    ax.plot(xs, ys, color=colors[source], marker=markers[source],
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

suptitle = f'Goal: "{goal_text}"' if goal_text else None
if "sys2_inference_ms" in df.columns:
    ms = df["sys2_inference_ms"].dropna()
    if not ms.empty and ms.iloc[0] > 0:
        detail = f"sys2 inference: {ms.iloc[0]:.0f} ms"
        suptitle = f"{suptitle}    ({detail})" if suptitle else detail

if suptitle:
    fig.suptitle(suptitle, fontsize=13)

fig.tight_layout()

out_path = data_dir / (csv_path.stem + f"_seq{seq_num}_plot.png")
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
