# generator.py


from __future__ import annotations

import os
import re

from .assets import load_world_config
from .layout import generate_layout, Placement


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


def generate_world_file(config_path, count, seed):

    config = load_world_config(config_path)

    # create path to world template
    package_dir = os.path.dirname(os.path.abspath(__file__))
    worlds_dir = os.path.normpath(os.path.join(package_dir, "..", "worlds"))
    template_world_path = os.path.join(worlds_dir, config.room.template_world)

    print(f"Using worlds dir: {worlds_dir}")
    print(f"Using template: {template_world_path}")

    # check template world exists
    if not os.path.exists(template_world_path):
        raise FileNotFoundError(f"Template world not found: {template_world_path}")

    placements = generate_layout(
        room=config.room,
        objects=config.objects,
        count=count,
        seed=seed,
    )

    include_blocks = [
        format_include_block(p, i)
        for i, p in enumerate(placements)
    ]

    with open(template_world_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    generated_text = insert_includes_into_world(template_text, include_blocks)

    output_path = os.path.join(worlds_dir, "generated_world.sdf")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(generated_text)


    print(f"Generated world written to: {output_path}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Placed {len(placements)} objects.")

    return output_path