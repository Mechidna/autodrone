#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import secrets
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PX4_WORLDS = Path(
    "/home/paolo/PX4-Autopilot/PX4-Autopilot/Tools/simulation/gz/worlds"
)
SRC = PX4_WORLDS / "gate_test_1500mm_blue.sdf"
OUTPUT_WORLD_NAME = "gate_test_1500mm_blue_random"
OUTPUT = PX4_WORLDS / f"{OUTPUT_WORLD_NAME}.sdf"

START_XY = (0.0, 0.0)
CAMERA_HFOV_MARGIN_DEG = 35.0
OUTER_GATE_HALF_WIDTH_M = 1.35
GATE_CENTER_Z_M = 1.35
GATE_ROOT_Z_RANGE_M = (0.0, 0.4)


@dataclass(frozen=True)
class RandomizationOptions:
    randomize_positions: bool = True
    randomize_gate_height: bool = False
    randomize_lighting: bool = False
    add_distractors: bool = False
    add_obstacles: bool = False
    randomize_gate_material: bool = False


@dataclass(frozen=True)
class WorldRandomizationResult:
    poses: list[tuple[float, float, float, float, float, float]]
    lighting: dict[str, float] | None
    gate_material: dict[str, float] | None
    distractor_count: int
    obstacle_count: int


def _bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str) -> None:
    parser.add_argument(
        name,
        type=_bool_arg,
        nargs="?",
        const=True,
        default=default,
        metavar="true|false",
        help=f"{help} Default: {str(default).lower()}.",
    )


def _visible_from_start(x_m: float, y_m: float) -> bool:
    dx_m = x_m - START_XY[0]
    dy_m = y_m - START_XY[1]
    if dx_m <= 0.0:
        return False

    lateral_limit_m = math.tan(math.radians(CAMERA_HFOV_MARGIN_DEG)) * dx_m
    return abs(dy_m) + OUTER_GATE_HALF_WIDTH_M < lateral_limit_m


def _random_gate_root_z(rng: random.Random, enabled: bool) -> float:
    if not enabled:
        return 0.0
    return rng.uniform(*GATE_ROOT_Z_RANGE_M)


def _generate_gate_poses(
    rng: random.Random,
    *,
    randomize_gate_height: bool,
) -> list[tuple[float, float, float, float, float, float]]:
    poses = []
    prev_x_m, prev_y_m = START_XY

    # SDF world frame for this PX4 world: x forward from start, y lateral, z up.
    # Keep z=0 because the gate model defines its own frame height.
    gate_ranges = (
        # first gate: absolute x range, y half-range
        (6.0, 10.0, 1.2),
        # later gates: delta-x range from previous gate, y half-range
        (6.0, 9.0, 2.0),
        (6.0, 9.0, 3.0),
    )

    for gate_idx, (x_min_m, x_max_m, y_limit_m) in enumerate(gate_ranges):
        for _ in range(1000):
            if gate_idx == 0:
                x_m = rng.uniform(x_min_m, x_max_m)
            else:
                x_m = prev_x_m + rng.uniform(x_min_m, x_max_m)
            y_m = rng.uniform(-y_limit_m, y_limit_m)

            if not _visible_from_start(x_m, y_m):
                continue

            yaw_rad = math.atan2(y_m - prev_y_m, x_m - prev_x_m)
            z_m = _random_gate_root_z(rng, randomize_gate_height)
            poses.append((x_m, y_m, z_m, 0.0, 0.0, yaw_rad))
            prev_x_m, prev_y_m = x_m, y_m
            break
        else:
            raise RuntimeError(f"Could not generate visible pose for gate {gate_idx + 1}.")

    return poses


def _parse_pose(text: str | None, *, context: str) -> tuple[float, float, float, float, float, float]:
    if text is None:
        raise RuntimeError(f"{context} has no pose text.")
    try:
        values = tuple(float(part) for part in text.split())
    except ValueError as exc:
        raise RuntimeError(f"{context} has invalid pose text: {text!r}") from exc
    if len(values) != 6:
        raise RuntimeError(f"{context} must contain 6 pose values, got {len(values)}.")
    return values


def _find_world(root: ET.Element) -> ET.Element:
    world = root.find("world")
    if world is None:
        raise RuntimeError(f"No <world> element found in {SRC}.")
    return world


def _find_gate_model(world: ET.Element, gate_idx: int) -> ET.Element:
    model = world.find(f"./model[@name='racing_gate_{gate_idx}']")
    if model is None:
        raise RuntimeError(f"No model named racing_gate_{gate_idx} found in {SRC}.")
    return model


def _get_gate_pose(world: ET.Element, gate_idx: int) -> tuple[float, float, float, float, float, float]:
    model = _find_gate_model(world, gate_idx)
    pose = model.find("pose")
    if pose is None:
        raise RuntimeError(f"racing_gate_{gate_idx} has no root <pose> element.")
    return _parse_pose(pose.text, context=f"racing_gate_{gate_idx}")


def _set_gate_pose(world: ET.Element, gate_idx: int, pose_values: tuple[float, ...]) -> None:
    model = _find_gate_model(world, gate_idx)
    pose = model.find("pose")
    if pose is None:
        raise RuntimeError(f"racing_gate_{gate_idx} has no root <pose> element.")
    pose.text = " ".join(f"{value:.6f}" for value in pose_values)


def _pilot_neu_center_from_sdf_pose(pose_values: tuple[float, ...]) -> tuple[float, float, float]:
    # Existing pilot logs/config use NEU-like gate centers where SDF x maps to forward/N,
    # and SDF y maps to lateral/E.
    sdf_x_m, sdf_y_m, sdf_z_m = pose_values[0], pose_values[1], pose_values[2]
    return (sdf_y_m, sdf_x_m, GATE_CENTER_Z_M + sdf_z_m)


def _set_text(parent: ET.Element, path: str, text: str) -> None:
    element = parent.find(path)
    if element is None:
        raise RuntimeError(f"Missing required SDF element: {path}")
    element.text = text


def _rgba(r: float, g: float, b: float, a: float = 1.0) -> str:
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"


def _randomize_lighting(world: ET.Element, rng: random.Random) -> dict[str, float]:
    light = world.find("./light[@name='sunUTC']")
    if light is None:
        raise RuntimeError("No light named sunUTC found in template world.")

    intensity = rng.uniform(0.70, 1.30)
    direction = (
        rng.uniform(-0.25, 0.25),
        rng.uniform(0.35, 0.85),
        rng.uniform(-0.95, -0.55),
    )
    norm = math.sqrt(sum(value * value for value in direction))
    direction = tuple(value / norm for value in direction)

    diffuse = rng.uniform(0.72, 1.00)
    specular = rng.uniform(0.18, 0.38)
    ambient = rng.uniform(0.28, 0.55)
    background = rng.uniform(0.55, 0.82)

    _set_text(light, "intensity", f"{intensity:.6f}")
    _set_text(
        light,
        "direction",
        f"{direction[0]:.6f} {direction[1]:.6f} {direction[2]:.6f}",
    )
    _set_text(light, "diffuse", _rgba(diffuse, diffuse, diffuse))
    _set_text(light, "specular", _rgba(specular, specular, specular))
    _set_text(world, "scene/ambient", _rgba(ambient, ambient, ambient))
    _set_text(world, "scene/background", _rgba(background, background, background))

    return {
        "intensity": intensity,
        "dir_x": direction[0],
        "dir_y": direction[1],
        "dir_z": direction[2],
        "ambient": ambient,
        "background": background,
    }


def _randomize_gate_material(world: ET.Element, rng: random.Random) -> dict[str, float]:
    blue = max(0.35, min(1.0, 0.701961 * rng.uniform(0.75, 1.15)))
    specular = rng.uniform(0.03, 0.15)

    for gate_idx in range(1, 4):
        model = _find_gate_model(world, gate_idx)
        for material in model.findall(".//material"):
            ambient = material.find("ambient")
            diffuse = material.find("diffuse")
            spec = material.find("specular")
            if ambient is not None:
                ambient.text = _rgba(0.0, 0.0, blue)
            if diffuse is not None:
                diffuse.text = _rgba(0.0, 0.0, blue)
            if spec is not None:
                spec.text = _rgba(specular, specular, specular)

    return {"blue": blue, "specular": specular}


def _append_text(parent: ET.Element, tag: str, text: str, **attrs: str) -> ET.Element:
    child = ET.SubElement(parent, tag, attrs)
    child.text = text
    return child


def _append_material(visual: ET.Element, rgba: tuple[float, float, float, float]) -> None:
    material = ET.SubElement(visual, "material")
    _append_text(material, "ambient", _rgba(*rgba))
    _append_text(material, "diffuse", _rgba(*rgba))
    _append_text(material, "specular", _rgba(0.08, 0.08, 0.08))


def _append_box_model(
    world: ET.Element,
    *,
    name: str,
    pose: tuple[float, float, float, float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    collision: bool,
) -> None:
    model = ET.SubElement(world, "model", {"name": name})
    _append_text(model, "static", "true")
    _append_text(model, "pose", " ".join(f"{value:.6f}" for value in pose))

    link = ET.SubElement(model, "link", {"name": "link"})
    if collision:
        collision_element = ET.SubElement(link, "collision", {"name": "collision"})
        collision_geometry = ET.SubElement(collision_element, "geometry")
        collision_box = ET.SubElement(collision_geometry, "box")
        _append_text(collision_box, "size", " ".join(f"{value:.6f}" for value in size))

    visual = ET.SubElement(link, "visual", {"name": "visual"})
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    _append_text(box, "size", " ".join(f"{value:.6f}" for value in size))
    _append_material(visual, rgba)


def _add_distractors(world: ET.Element, rng: random.Random) -> int:
    count = rng.randint(2, 5)
    for idx in range(count):
        size = (
            rng.uniform(0.35, 1.25),
            rng.uniform(0.35, 1.25),
            rng.uniform(0.40, 1.80),
        )
        side = -1.0 if rng.random() < 0.5 else 1.0
        pose = (
            rng.uniform(5.0, 25.0),
            side * rng.uniform(4.0, 8.0),
            size[2] * 0.5,
            0.0,
            0.0,
            rng.uniform(-math.pi, math.pi),
        )
        shade = rng.uniform(0.22, 0.55)
        _append_box_model(
            world,
            name=f"random_distractor_{idx + 1}",
            pose=pose,
            size=size,
            rgba=(shade, shade, shade * rng.uniform(0.85, 1.10), 1.0),
            collision=False,
        )
    return count


def _add_obstacles(world: ET.Element, rng: random.Random, gate_poses: list[tuple[float, ...]]) -> int:
    count = 4
    xs = [pose[0] for pose in gate_poses]
    min_x = max(4.0, min(xs) - 1.5)
    max_x = max(xs) + 1.5

    for idx in range(count):
        side = -1.0 if idx % 2 == 0 else 1.0
        height = rng.uniform(2.2, 3.2)
        size = (rng.uniform(0.18, 0.35), rng.uniform(0.18, 0.35), height)
        pose = (
            rng.uniform(min_x, max_x),
            side * rng.uniform(5.0, 7.0),
            height * 0.5,
            0.0,
            0.0,
            rng.uniform(-0.2, 0.2),
        )
        _append_box_model(
            world,
            name=f"random_boundary_post_{idx + 1}",
            pose=pose,
            size=size,
            rgba=(0.20, 0.20, 0.20, 1.0),
            collision=True,
        )
    return count


def _write_random_world(seed: int, options: RandomizationOptions) -> WorldRandomizationResult:
    tree = ET.parse(SRC)
    root = tree.getroot()
    world = _find_world(root)
    world.set("name", OUTPUT_WORLD_NAME)

    rng = random.Random(seed)
    if options.randomize_positions:
        poses = _generate_gate_poses(
            rng,
            randomize_gate_height=options.randomize_gate_height,
        )
    else:
        poses = [_get_gate_pose(world, gate_idx) for gate_idx in range(1, 4)]
        if options.randomize_gate_height:
            poses = [
                (
                    pose[0],
                    pose[1],
                    _random_gate_root_z(rng, enabled=True),
                    pose[3],
                    pose[4],
                    pose[5],
                )
                for pose in poses
            ]

    for gate_idx, pose_values in enumerate(poses, start=1):
        _set_gate_pose(world, gate_idx, pose_values)

    lighting = _randomize_lighting(world, rng) if options.randomize_lighting else None
    gate_material = (
        _randomize_gate_material(world, rng)
        if options.randomize_gate_material
        else None
    )
    distractor_count = _add_distractors(world, rng) if options.add_distractors else 0
    obstacle_count = _add_obstacles(world, rng, poses) if options.add_obstacles else 0

    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="UTF-8", xml_declaration=True)
    return WorldRandomizationResult(
        poses=poses,
        lighting=lighting,
        gate_material=gate_material,
        distractor_count=distractor_count,
        obstacle_count=obstacle_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one overwritten randomized PX4 Gazebo gate world from "
            "gate_test_1500mm_blue.sdf."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducing a generated gate layout.",
    )
    _add_bool_arg(
        parser,
        "--randomize-positions",
        default=True,
        help="Randomize gate x/y positions and gate yaw while keeping gates in startup camera view.",
    )
    _add_bool_arg(
        parser,
        "--randomize-gate-height",
        default=False,
        help="Randomize gate root z slightly without changing gate dimensions.",
    )
    _add_bool_arg(
        parser,
        "--randomize-lighting",
        default=False,
        help="Randomize sun direction/intensity and scene ambient/background values.",
    )
    _add_bool_arg(
        parser,
        "--add-distractors",
        default=False,
        help="Add non-gate visual static scene objects outside the direct racing line.",
    )
    _add_bool_arg(
        parser,
        "--add-obstacles",
        default=False,
        help="Add boundary-style collision posts near, but outside, the racing line.",
    )
    _add_bool_arg(
        parser,
        "--randomize-gate-material",
        default=False,
        help="Apply one subtle material brightness variation uniformly to all gates.",
    )
    args = parser.parse_args()

    seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    options = RandomizationOptions(
        randomize_positions=bool(args.randomize_positions),
        randomize_gate_height=bool(args.randomize_gate_height),
        randomize_lighting=bool(args.randomize_lighting),
        add_distractors=bool(args.add_distractors),
        add_obstacles=bool(args.add_obstacles),
        randomize_gate_material=bool(args.randomize_gate_material),
    )
    result = _write_random_world(seed, options)

    print(f"seed={seed}")
    print(f"template={SRC}")
    print(f"wrote={OUTPUT}")
    print(f"launch=PX4_GZ_WORLD={OUTPUT_WORLD_NAME} make px4_sitl gz_x500_mono_cam")
    print(
        "options="
        f"positions={int(options.randomize_positions)} "
        f"gate_height={int(options.randomize_gate_height)} "
        f"lighting={int(options.randomize_lighting)} "
        f"distractors={int(options.add_distractors)} "
        f"obstacles={int(options.add_obstacles)} "
        f"gate_material={int(options.randomize_gate_material)}"
    )
    if result.lighting is not None:
        print(
            "lighting "
            f"intensity={result.lighting['intensity']:.3f} "
            f"direction=({result.lighting['dir_x']:.3f},"
            f"{result.lighting['dir_y']:.3f},{result.lighting['dir_z']:.3f}) "
            f"ambient={result.lighting['ambient']:.3f} "
            f"background={result.lighting['background']:.3f}"
        )
    if result.gate_material is not None:
        print(
            "gate_material "
            f"blue={result.gate_material['blue']:.3f} "
            f"specular={result.gate_material['specular']:.3f}"
        )
    if result.distractor_count:
        print(f"distractors={result.distractor_count}")
    if result.obstacle_count:
        print(f"obstacles={result.obstacle_count}")
    for gate_idx, pose_values in enumerate(result.poses, start=1):
        pilot_neu = _pilot_neu_center_from_sdf_pose(pose_values)
        print(
            f"gate_{gate_idx} "
            f"sdf_pose=({pose_values[0]:.3f},{pose_values[1]:.3f},"
            f"{pose_values[2]:.3f},{pose_values[3]:.3f},"
            f"{pose_values[4]:.3f},{pose_values[5]:.3f}) "
            f"approx_pilot_neu_center=({pilot_neu[0]:.3f},"
            f"{pilot_neu[1]:.3f},{pilot_neu[2]:.3f})"
        )


if __name__ == "__main__":
    main()
