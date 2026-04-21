# generator.py

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from assets import load_world_config
from layout import generate_layout, Placement


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Gazebo world with random object placement.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--count", type=int, required=True, help="Number of objects to place.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible layouts.")
    args = parser.parse_args()

    config = load_world_config(args.config)

    # create path to world template
    script_path = Path(__file__).resolve()
    worlds_dir = script_path.parent.parent  # object_spawner -> worlds
    template_world_path = worlds_dir / config.room.template_world

    print(f"Using worlds dir: {worlds_dir}")
    print(f"Using template: {template_world_path}")

    # check template world exists
    if not template_world_path.exists():
        raise FileNotFoundError(f"Template world not found: {template_world_path}")

    placements = generate_layout(
        room=config.room,
        objects=config.objects,
        count=args.count,
        seed=args.seed,
    )

    include_blocks = [
        format_include_block(p, i)
        for i, p in enumerate(placements)
    ]

    template_text = template_world_path.read_text(encoding="utf-8")
    generated_text = insert_includes_into_world(template_text, include_blocks)

    output_path = worlds_dir / "generated_world.sdf"
    output_path.write_text(generated_text, encoding="utf-8")

    print(f"Generated world written to: {output_path}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"Placed {len(placements)} objects.")


if __name__ == "__main__":
    main()