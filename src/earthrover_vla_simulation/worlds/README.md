To open unempty worlds, or any worlds that include local models:

```bash
# Go to the repository root
cd /vla-autonomous-vehicles

# Set Gazebo's resource path so local models can be found
export GZ_SIM_RESOURCE_PATH=$PWD/src/earthrover_vla_simulation/worlds/custom_worlds/local_models

# Launch Gazebo with the desired world
gz sim $PWD/src/earthrover_vla_simulation/worlds/unempty_office_square.sdf
