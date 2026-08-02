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

Two vision-language-action models run as ROS 2 inference nodes:

- **AsyncVLA** (`async_vla`) — `sys2` backbone VLA + `sys1` edge adapter
- **TIC-VLA** (`tic_vla`) — `image_processing` + `sys2` VLM backbone + `sys1` action expert

This workspace holds the ROS 2 wrappers. The model code lives in separate upstream repos,
cloned alongside it.

Both models publish to the same controller topics, so run one at a time.

## Machines

Selected at launch time with `device:=`:

| `device:=` | Machine |
| --- | --- |
| `dsk` | Lab desktop (`vla-cap` user), on the robot's local network |
| `rcp` | Remote uni desktop, reached over SSH; starts its own zenoh router |

Two launch files per package:

```
async_vla/launch/asc_async.launch.py   tic_vla/launch/asc_tic.launch.py    # inference nodes
async_vla/launch/sim_async.launch.py   tic_vla/launch/sim_tic.launch.py    # inference + Gazebo + control
```

Each package has one copy of each node source file. Model paths come from a `DEVICE_PATHS`
table at the top of each launch file and reach the nodes as ROS parameters. Adding a machine
means adding a row to that table. An unrecognised `device:=` reports the valid options and
stops.

The `scripts/` wrappers come in `_rcp`/`_dsk` pairs, selected by `device:=`. They set the
conda interpreter and `sys.path`, which Python resolves at import time, ahead of ROS
parameters.

## 1. Clone The Model Repos

```bash
git clone <AsyncVLA repo> AsyncVLA
cd AsyncVLA && git submodule update --init --recursive

git clone <TIC-VLA repo> TIC-VLA
cd TIC-VLA && git submodule update --init --recursive
```

Locations, matching `DEVICE_PATHS` in the launch files:

| | rcp | dsk |
| --- | --- | --- |
| AsyncVLA repo | `/vla_storage/capstone/code/asyncvla/AsyncVLA` | `/home/vla-cap/AsyncVLA` |
| AsyncVLA weights | `…/AsyncVLA/AsyncVLA_release` | `/home/vla-cap/AsyncVLA/AsyncVLA_release` |
| TIC-VLA repo | `/vla_storage/capstone/code/ticvla/TIC-VLA` | `/home/vla-cap/capstone/code/ticvla/TIC-VLA` |
| InternVL3-1B | `/vla_storage/capstone/code/ticvla/InternVL3-1B` | `/home/vla-cap/capstone/code/ticvla/InternVL3-1B` |
| TIC-VLA ckpt | `/vla_storage/capstone/code/ticvla/TIC-VLA-model.ckpt` | `/home/vla-cap/capstone/code/ticvla/TIC-VLA-model.ckpt` |

On rcp the weights live on `/vla_storage`, which has room for them.

## 2. Run The Setup Scripts

Each script initialises submodules, installs Miniconda if missing, creates the conda env,
installs the Python package, and downloads the weights.

dsk:

```bash
bash tools/setup_asyncvla_desktop.bash
bash tools/setup_ticvla_desktop.bash
```

rcp:

```bash
ASYNCVLA_BASE_DIR=/vla_storage/capstone/code/asyncvla bash tools/setup_asyncvla_desktop.bash
TICVLA_BASE_DIR=/vla_storage/capstone/code/ticvla   bash tools/setup_ticvla_desktop.bash
```

Both are idempotent — re-run to resume an interrupted download.

The scripts create two conda envs, `asyncvla` and `tic-vla`, both on Python 3.10, matching
the Python that ROS 2 Humble's `rclpy` C extension is built for.

Weights come from `NHirose/AsyncVLA_release`, `OpenGVLab/InternVL3-1B`, and
`handsomeYun/TIC-VLA` (a dataset repo, file `TIC-VLA-model.ckpt`).

## 3. Build The Workspace

`custom_msgs` and `earthrover_vla_simulation` generate Python bindings against the active
interpreter, so build them with conda deactivated to get ROS Humble's Python 3.10:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Build with `--symlink-install` throughout, so colcon's symlinks stay consistent across
packages.

`--symlink-install` also makes edits to node `.py` files and launch files take effect
directly. Rebuild after adding files to `scripts/` or `launch/`, or changing `setup.py`,
`package.xml`, or `custom_msgs`.

After rebuilding, source `install/setup.bash` in a new shell to pick up the current paths.

## 4. Networking (rcp)

dsk reaches the robot over the local network.

For rcp, open the tunnel and leave it running:

```bash
ssh -L 7447:localhost:7447 <rcp address>
```

Edit both zenoh configs on both machines (root-owned, needs `sudo`):

`/opt/ros/humble/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5`

```json5
listen: {
  endpoints: ["tcp/localhost:7447"],
},
```

`/opt/ros/humble/share/rmw_zenoh_cpp/config/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5`

```json5
mode: "client",
connect: {
  endpoints: ["tcp/localhost:7447"],
},
```

`rmw_zenoh_cpp` routes through a `rmw_zenohd` process. With `device:=rcp` the launch files
start one if none is running, set `RMW_IMPLEMENTATION` and `ROS_DOMAIN_ID`, and shut it down
on Ctrl-C. To keep a router up across launches, start one yourself:

```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

## 5. Launching

### Sim on one machine

Gazebo, control nodes and inference together:

```bash
ros2 launch async_vla sim_async.launch.py
ros2 launch tic_vla   sim_tic.launch.py
```

### Sim split across two machines

Sim and control on the VM:

```bash
ros2 launch asc asc_sim.launch.py
```

Inference on rcp. Gazebo publishes `sensor_msgs/Image` on `/cam`, which `use_sim:=true`
selects:

```bash
ros2 launch async_vla asc_async.launch.py device:=rcp use_sim:=true
```

### Hardware

Control stack on the robot:

```bash
ros2 launch asc asc.launch.py
```

Inference on whichever machine. The camera publishes `ImageWithSeqNum` on `/cam`, which is
the default:

```bash
ros2 launch async_vla asc_async.launch.py device:=rcp
ros2 launch async_vla asc_async.launch.py device:=dsk
ros2 launch tic_vla   asc_tic.launch.py   device:=dsk
```

### Arguments

```bash
ros2 launch async_vla asc_async.launch.py goal:="Find the red door"
ros2 launch async_vla asc_async.launch.py device:=rcp domain_id:=7
ros2 launch tic_vla   sim_tic.launch.py   worldfile:=unempty_office_square.sdf
ros2 launch async_vla asc_async.launch.py --show-args
```

`goal:=` applies to all four launch files and both models, and is read once at startup.
The launch file supplies its value to the nodes.

| Launch file | Arguments |
| --- | --- |
| `asc_async.launch.py`, `asc_tic.launch.py` | `device`, `use_sim`, `goal`, `domain_id` |
| `sim_async.launch.py`, `sim_tic.launch.py` | `device`, `worldfile`, `goal` |
