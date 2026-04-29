# generator.py


from __future__ import annotations

import os
import re

from ament_index_python.packages import get_package_share_directory
from .assets import load_world_config
from .layout import generate_layout, Placement
import time
import random
import glob


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def format_include_block(placement: Placement, index: int) -> str:
    instance_name = f"{sanitize_name(placement.object_name)}_{index}"
    return f"""    <include>
      <uri>{placement.uri}</uri>
      <name>{instance_name}</name>
      <pose>{placement.x:.6f} {placement.y:.6f} {placement.z:.6f} 0 0 {placement.yaw:.6f}</pose>
    </include>"""


def insert_includes_into_world(world_text: str, include_blocks: list[str]) -> str:
    insert_text = "\n".join(include_blocks) + "\n"
    marker = "</world>"

    idx = world_text.rfind(marker)
    if idx == -1:
        raise ValueError("Could not find </world> in template world file.")

    return world_text[:idx] + insert_text + world_text[idx:]


# Accepts seed, randomly generates object count,
# selects yaml file that defines the world template to generate from
def generate_world_file():
    # Define relevant directories
    pkg_dir = get_package_share_directory('earthrover_vla_simulation')
    worlds_dir = os.path.join(pkg_dir, 'worlds')
    templates_dir=os.path.join(worlds_dir, 'templates')
    gen_configs_dir = os.path.join(worlds_dir,'generation_configs')
    log_path = os.path.join(worlds_dir, 'autogen_log.txt')
    
    # Set randomiser seed
    seed = int(time.time()*1000) #Generates seed from current time to ms resolution
    rng = random.Random(seed)

    # Randomly select a config .yaml file to define the world template
    avail_configs = sorted(glob.glob(os.path.join(gen_configs_dir,"*.yaml")))
    if not avail_configs:                                                                                                    
        raise RuntimeError(f"No config files found in {gen_configs_dir}")
    config_path = rng.choice(avail_configs)
    config = load_world_config(config_path)
    template_world_path = os.path.join(templates_dir ,config.room.template_world)

    # Randomise number of objects to spawn
    obj_count = rng.randint(3,5)

    # check template world exists
    if not os.path.exists(template_world_path):
        raise FileNotFoundError(f"Template world not found: {template_world_path}")

    placements = generate_layout(
        room=config.room,
        objects=config.objects,
        count=obj_count,
        seed=seed,
    )

    include_blocks = [
        format_include_block(p, i)
        for i, p in enumerate(placements)
    ]

    with open(template_world_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    generated_text = insert_includes_into_world(template_text, include_blocks)
    generated_text = re.sub(r"<world name='[^']*'>", "<world name='generated_world'>", generated_text, count=1)

    output_path = os.path.join(templates_dir, "generated_world.sdf")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_text)

    with open(log_path, 'a') as f:                                                                                         
        f.write(f"{time.strftime("%Y-%m-%d %H:%M:%S")} | seed={seed} | config={os.path.basename(config_path)} | objects={obj_count}\n")

    return "generated_world.sdf"