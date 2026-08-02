# Earthrover VLA Workspace

This workspace is split into separate ROS 2 packages for robot description, simulation, and top-level bringup.

## Workspace Architecture

The `src/` folder currently contains three packages:

- `earthrover_vla_description`
  Stores the base robot model and package metadata for the physical platform.
  Package structure:
  `urdf/`
  Contains the core Xacro / URDF description of the robot, including links, joints, geometry, and inertial properties.

- `earthrover_vla_simulation`
  Stores simulation-specific assets, Gazebo configuration, and the launch logic needed to run the robot in Gazebo Sim.
  Package structure:
  `urdf/`
  Contains simulation-oriented robot description overlays, including the top-level simulation Xacro and Gazebo plugin / sensor extensions.
  `worlds/`
  Contains Gazebo world definitions used for simulation scenarios.
  `config/`
  Contains runtime configuration files, such as ROS-Gazebo bridge settings.
  `launch/`
  Contains launch files responsible for starting the simulation stack.

- `earthrover_vla_bringup`
  Provides the top-level entrypoints for launching the overall system and selecting between operating modes such as simulation and hardware.
  Package structure:
  `launch/`
  Contains wrapper launch files that route to the appropriate lower-level launch path for the chosen mode.

The dependency flow is:

`earthrover_vla_description` -> base robot model

`earthrover_vla_simulation` -> includes the description package and adds Gazebo-specific configuration

`earthrover_vla_bringup` -> includes the simulation launch file, and later can branch to hardware bringup

## Build And Source

From the workspace root:

```bash
cd ~/vla-capstone-ws
colcon build
source install/setup.bash
```

If you only want to rebuild these packages:

```bash
colcon build --packages-select \
  earthrover_vla_description \
  earthrover_vla_simulation \
  earthrover_vla_bringup
source install/setup.bash
```

After editing launch files, Xacro files, worlds, package metadata, or CMake install rules, rebuild and source again before testing.

## Clean Rebuild

If you want to force a fresh rebuild of the workspace:

```bash
rm -rf build install log
colcon build
source install/setup.bash
```

## Launching

The recommended entrypoint is the bringup package:

```bash
ros2 launch earthrover_vla_bringup launch.py
```

This defaults to simulation mode.

You can also launch the simulation package directly:

```bash
ros2 launch earthrover_vla_simulation sim.launch.py
```

## Sim Mode

To launch simulation explicitly through the bringup wrapper:

```bash
ros2 launch earthrover_vla_bringup launch.py hardware:=false
```

This will:

- expand `earthrover_vla_simulation/urdf/robot_sim.xacro`
- launch Gazebo Sim with `worlds/empty_world_cam.sdf`
- spawn the robot from `robot_description`
- start `robot_state_publisher`
- start the `ros_gz_bridge` parameter bridge

## Hardware Mode

Hardware mode is not implemented yet.

For now, this command just prints a TODO message:

```bash
ros2 launch earthrover_vla_bringup launch.py hardware:=true
```

## Notes

- The simulation Xacro uses package-based includes, so the workspace must be built and sourced before those lookups work.
- The world file currently used by simulation is `empty_world_cam.sdf`.
- The top-level bringup script is intentionally simple and acts as the mode selector for future sim and hardware launch paths.

---

# Running the VLA Models (`async_vla` / `tic_vla`)

This branch runs two vision-language-action models as ROS 2 inference nodes:

- **AsyncVLA** (`async_vla` package) — `sys2` backbone VLA + `sys1` edge adapter
- **TIC-VLA** (`tic_vla` package) — `image_processing` + `sys2` VLM backbone + `sys1` action expert

Both models live **outside** this workspace as separate upstream repos. This workspace only
contains the ROS 2 wrappers; you must clone and install the model repos separately.

## Machine Roles

Two GPU machines are supported, and most files come in matching pairs:

| Suffix | Machine | Notes |
| --- | --- | --- |
| `_dsk` | Lab desktop (`vla-cap` user) | Runs on the local network with the robot. |
| `_rcp` | Remote uni desktop | Reached over SSH; needs a tunnel to talk to the robot. |

The suffix appears on both the node wrapper scripts (`scripts/sys1_rcp`, `scripts/sys1_dsk`, …)
and the hardware launch files (`asc_async_rcp.launch.py`, `asc_async_dsk.launch.py`, …).

**The node source files are shared** — `async_vla/sys1.py`, `async_vla/sys2.py`,
`tic_vla/sys1.py`, `tic_vla/sys2.py` exist once each. All machine-specific paths are
injected by the launch files as ROS parameters (`vla_path`, `shead_path`, `vlm_path`,
`checkpoint_path`), so adding a new machine means adding a launch file, not a new node.

The wrapper scripts in `scripts/` still come in `_rcp`/`_dsk` pairs because the conda
interpreter (shebang) and `sys.path` must be set *before* Python imports `prismatic`/`ticvla`,
which is earlier than any ROS parameter can be read.

## 1. Clone The Model Repos

Clone these on whichever machine will run inference.

**AsyncVLA**

```bash
git clone <AsyncVLA repo> AsyncVLA
cd AsyncVLA && git submodule update --init --recursive
```

**TIC-VLA**

```bash
git clone <TIC-VLA repo> TIC-VLA
cd TIC-VLA && git submodule update --init --recursive
```

### Expected locations

Paths differ per machine. These are the values currently hardcoded in the launch files —
if you clone elsewhere, update the corresponding launch file.

| | rcp | dsk |
| --- | --- | --- |
| AsyncVLA repo | `/vla_storage/capstone/code/asyncvla/AsyncVLA` | `/home/vla-cap/AsyncVLA` |
| AsyncVLA weights | `…/AsyncVLA/AsyncVLA_release` | `/home/vla-cap/AsyncVLA/AsyncVLA_release` |
| TIC-VLA repo | `/vla_storage/capstone/code/ticvla/TIC-VLA` | `/home/vla-cap/capstone/code/ticvla/TIC-VLA` |
| InternVL3-1B | `/vla_storage/capstone/code/ticvla/InternVL3-1B` | `/home/vla-cap/capstone/code/ticvla/InternVL3-1B` |
| TIC-VLA ckpt | `/vla_storage/capstone/code/ticvla/TIC-VLA-model.ckpt` | `/home/vla-cap/capstone/code/ticvla/TIC-VLA-model.ckpt` |

> **rcp uses `/vla_storage`, not `$HOME`.** The home filesystem on the uni desktop is small
> and filled up completely during setup, which silently truncated model downloads. Keep the
> model repos and weights on `/vla_storage`.

## 2. Run The Setup Scripts

`tools/` contains a first-time setup script per model. Each one initialises submodules,
installs Miniconda if missing, creates the conda env, installs the Python package, and
downloads the model weights from HuggingFace.

On **dsk**, the defaults are correct:

```bash
bash tools/setup_asyncvla_desktop.bash
bash tools/setup_ticvla_desktop.bash
```

On **rcp**, point them at `/vla_storage` — the home filesystem is too small:

```bash
ASYNCVLA_BASE_DIR=/vla_storage/capstone/code/asyncvla bash tools/setup_asyncvla_desktop.bash
TICVLA_BASE_DIR=/vla_storage/capstone/code/ticvla   bash tools/setup_ticvla_desktop.bash
```

Both scripts are idempotent — re-run them to resume an interrupted weight download.

**Note on the TIC-VLA env:** upstream's `tic-vla.yaml` specifies Python 3.11, which cannot
be used here. ROS 2 Humble's `rclpy` C extension is built for Python 3.10 only, so a 3.11
env fails with `No module named 'rclpy._rclpy_pybind11'`. The setup script deliberately
ignores `tic-vla.yaml` and creates the env at 3.10, installing `requirements-train.txt`
into it instead.

Both scripts also install `opencv-python-headless==4.11.0.86` for `cv_bridge`. The pin
matters: opencv ≥ 4.12 requires `numpy>=2`, which conflicts with AsyncVLA's `numpy==1.26.4`
and TIC-VLA's `numpy>=1.26,<2.0`.

### Model weight downloads

The weights are large and the downloads are resumable — if one is interrupted, **re-run the
same command**, don't delete the directory first. Deleting discards the partial blobs in
`.cache/huggingface/download/` and forces a full re-download.

| Asset | Source |
| --- | --- |
| AsyncVLA release | `NHirose/AsyncVLA_release` (model repo) |
| InternVL3-1B | `OpenGVLab/InternVL3-1B` (model repo) |
| TIC-VLA checkpoint | `handsomeYun/TIC-VLA`, file `TIC-VLA-model.ckpt` (**dataset** repo) |

A truncated AsyncVLA download is easy to misdiagnose: a missing `tokenizer_config.json`
surfaces as `KeyError: 'OpenVLAConfig'` deep inside `transformers`, which looks like a
library version problem but isn't. Verify no `*.incomplete` files remain:

```bash
find <weights dir> -name '*.incomplete'
```

## 3. Build The Workspace

`custom_msgs` and `earthrover_vla_simulation` generate Python bindings, so they must be
built against the **same Python 3.10 that ROS Humble uses** — not conda base (often 3.13).
Deactivate conda before building them, or you'll get
`No module named 'custom_msgs.custom_msgs_s__rosidl_typesupport_c'` at runtime.

```bash
conda deactivate                 # repeat until no env is active
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

The `async_vla` and `tic_vla` packages install their node modules via an editable link back
to `src/`, so **editing a node's `.py` file takes effect without rebuilding**. You still need
to rebuild after changing launch files, wrapper scripts, or `setup.py`.

## 4. Networking (rcp only)

On **dsk** the machine is on the same network as the robot, so no tunnel is needed.

On **rcp** the inference nodes run on the remote uni desktop while the robot is elsewhere,
so traffic is carried over an SSH tunnel. Zenoh's router listens on TCP **7447**.

Open the tunnel from the local machine and leave it running for the whole session:

```bash
ssh -L 7447:localhost:7447 <rcp address>
```

This forwards port 7447 so both ends can reach the zenoh router over loopback.

### Zenoh config files

Both machines need the stock `rmw_zenoh_cpp` configs edited so the router listens on, and
sessions connect to, `localhost:7447` — matching the tunnel above. The same two edits apply
to **both** the rcp and the local machine.

`/opt/ros/humble/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5`

```json5
listen: {
  endpoints: ["tcp/localhost:7447"],
},
```

`/opt/ros/humble/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5`

```json5
mode: "client",   // "peer" also works when the router and node are on the same device
connect: {
  endpoints: ["tcp/localhost:7447"],
},
```

These files are root-owned, so edit them with `sudo`.

### Zenoh router

`rmw_zenoh_cpp` requires a standalone router process before any node can initialise;
without it nodes die with `Error setting up zenoh session`.

**The `_rcp` hardware launch files handle this for you.** They start `rmw_zenohd` if one
isn't already running, wait 2s for it to bind, and shut it down cleanly on Ctrl-C. They also
set `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and `ROS_DOMAIN_ID` for all child processes, so you
no longer need to export those by hand. Override the domain with:

```bash
ros2 launch async_vla asc_async_rcp.launch.py domain_id:=7
```

The `_dsk` and `sim_*` launch files do **not** do this — start the router yourself first:

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd &
```

## 5. Launch Matrix

Pick one row. Inference and control are separate launches and run in separate terminals
(and, for the rcp/hardware rows, on separate machines).

| Scenario | Inference launch | Control / sim launch |
| --- | --- | --- |
| **dsk, sim only** (single machine) | `ros2 launch async_vla sim_async.launch.py`<br>`ros2 launch tic_vla sim_tic.launch.py` | *(included — these launch Gazebo and the asc control nodes themselves)* |
| **rcp, sim** | `ros2 launch async_vla asc_async_rcp.launch.py`<br>`ros2 launch tic_vla asc_tic_rcp.launch.py` | `ros2 launch asc asc_sim.launch.py` |
| **Hardware deployment** | `ros2 launch async_vla asc_async_rcp.launch.py`<br>*or* `…_dsk.launch.py`<br>`ros2 launch tic_vla asc_tic_rcp.launch.py`<br>*or* `…_dsk.launch.py` | `ros2 launch asc asc.launch.py` *(on the robot)* |

Run **either** the AsyncVLA launch **or** the TIC-VLA launch — not both at once; they both
drive the same controller topics.

### What each launch file contains

- `sim_async.launch.py` / `sim_tic.launch.py` — self-contained single-machine sim: Gazebo
  (via `earthrover_vla_bringup`), the `asc` control nodes in sim mode, and the inference
  nodes. Accepts `worldfile:=` and a goal/instruction argument.
- `asc_async_{rcp,dsk}.launch.py` / `asc_tic_{rcp,dsk}.launch.py` — inference nodes only.
  Pair with an `asc` launch for control.
- `asc_sim.launch.py` — Gazebo + `asc` control nodes, no inference.
- `asc.launch.py` — full hardware control stack: camera capture, encoder odometry, outer/inner
  loop controllers, and the Roboclaw motor driver. Runs on the robot.

### Setting the language goal

Every launch file — sim and hardware, both models — takes the same `goal:=` argument:

```bash
ros2 launch async_vla asc_async_rcp.launch.py goal:="Find the red door"
ros2 launch tic_vla   asc_tic_rcp.launch.py   goal:="Find the red door"
ros2 launch async_vla sim_async.launch.py     goal:="Find the red door"
```

Omit it to use the launch file's default (`Go to the yellow bin`). The goal is only read at
startup, so changing it means restarting the launch.

The nodes themselves have **no built-in default** — the goal comes entirely from the launch
file, the same way the model paths do. TIC-VLA's node parameter is still called
`instruction` internally (that's the term the VLM prompt uses); the launch files map
`goal` → `instruction` so the command line stays uniform across both models.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Error setting up zenoh session` | No `rmw_zenohd` running. Use an `_rcp` launch, or start it manually. |
| `KeyError: 'OpenVLAConfig'` | Truncated AsyncVLA download — `tokenizer_config.json` missing. Re-run the download. |
| `No module named 'custom_msgs.custom_msgs_s__rosidl_typesupport_c'` | `custom_msgs` built under the wrong Python. Rebuild with conda deactivated. |
| `No module named 'rclpy._rclpy_pybind11'` | Conda env is not Python 3.10. Recreate it at 3.10. |
| `No module named 'cv2'` | `pip install "opencv-python-headless<5"` into the env. |
| `executable '…' not found on the libexec directory` | Wrapper script in `scripts/` is missing its executable bit, or the workspace wasn't rebuilt after adding it. |
| Model downloads fail immediately / produce empty files | Filesystem full. Check `df -h`; on rcp keep everything on `/vla_storage`. |
