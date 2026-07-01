#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import copy
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
RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / "config" / "runtime.toml"

DEFAULT_GATE_COUNT = 3
DEFAULT_GATE_SPACING_M = 20
START_XY = (0.0, 0.0)
CAMERA_HFOV_MARGIN_DEG = 35.0
OUTER_GATE_HALF_WIDTH_M = 1.35
GATE_CENTER_Z_M = 1.35
GATE_ROOT_Z_RANGE_M = (0.0, 0.4)
SEQUENTIAL_FIRST_GATE_ROOT_Z_MAX_M = 5.0
FIXED_GATE_RGB_SPECULAR = 0.08


@dataclass(frozen=True)
class RandomizationOptions:
    gate_count: int = DEFAULT_GATE_COUNT
    gate_spacing_m: int = DEFAULT_GATE_SPACING_M
    randomize_positions: bool = True
    randomize_gate_height: bool = False
    sequential_gate_height_step_m: float | None = None
    randomize_lighting: bool = False
    add_distractors: bool = False
    add_obstacles: bool = False
    randomize_gate_material: bool = False
    randomize_gate_color: bool = False
    gate_rgb: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class WorldRandomizationResult:
    poses: list[tuple[float, float, float, float, float, float]]
    lighting: dict[str, float] | None
    gate_material: dict[str, float] | None
    gate_color: dict[str, float] | None
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


def _random_sequential_gate_root_zs(
    rng: random.Random,
    *,
    gate_count: int,
    step_m: float,
) -> list[float]:
    heights: list[float] = []
    for gate_idx in range(int(gate_count)):
        if gate_idx == 0:
            min_z_m = 0.0
            max_z_m = SEQUENTIAL_FIRST_GATE_ROOT_Z_MAX_M
        else:
            prev_z_m = heights[-1]
            min_z_m = max(0.0, prev_z_m - float(step_m))
            max_z_m = prev_z_m + float(step_m)
        heights.append(rng.uniform(min_z_m, max_z_m))
    return heights


def _random_gate_root_zs(
    rng: random.Random,
    *,
    gate_count: int,
    randomize_gate_height: bool,
    sequential_gate_height_step_m: float | None,
) -> list[float] | None:
    if sequential_gate_height_step_m is not None:
        return _random_sequential_gate_root_zs(
            rng,
            gate_count=gate_count,
            step_m=float(sequential_gate_height_step_m),
        )
    if randomize_gate_height:
        return [_random_gate_root_z(rng, enabled=True) for _ in range(int(gate_count))]
    return None


def _replace_gate_root_zs(
    poses: list[tuple[float, float, float, float, float, float]],
    root_zs_m: list[float] | None,
) -> list[tuple[float, float, float, float, float, float]]:
    if root_zs_m is None:
        return poses
    return [
        (pose[0], pose[1], float(z_m), pose[3], pose[4], pose[5])
        for pose, z_m in zip(poses, root_zs_m, strict=True)
    ]


def _generate_gate_poses(
    rng: random.Random,
    *,
    gate_count: int,
    gate_spacing_m: int,
    randomize_gate_height: bool,
    sequential_gate_height_step_m: float | None,
) -> list[tuple[float, float, float, float, float, float]]:
    poses = []
    prev_x_m, prev_y_m = START_XY

    # SDF world frame for this PX4 world: x forward from start, y lateral, z up.
    # Keep z=0 because the gate model defines its own frame height.
    spacing_m = float(gate_spacing_m)
    spacing_jitter_m = max(0.25, spacing_m * 0.10)

    for gate_idx in range(int(gate_count)):
        y_limit_m = min(4.0, 1.2 + 0.45 * gate_idx)
        for _ in range(1000):
            forward_step_m = rng.uniform(
                spacing_m - spacing_jitter_m,
                spacing_m + spacing_jitter_m,
            )
            x_m = prev_x_m + max(2.0, forward_step_m)
            y_m = rng.uniform(-y_limit_m, y_limit_m)

            if not _visible_from_start(x_m, y_m):
                continue

            yaw_rad = math.atan2(y_m - prev_y_m, x_m - prev_x_m)
            poses.append((x_m, y_m, 0.0, 0.0, 0.0, yaw_rad))
            prev_x_m, prev_y_m = x_m, y_m
            break
        else:
            raise RuntimeError(f"Could not generate visible pose for gate {gate_idx + 1}.")

    root_zs_m = _random_gate_root_zs(
        rng,
        gate_count=gate_count,
        randomize_gate_height=randomize_gate_height,
        sequential_gate_height_step_m=sequential_gate_height_step_m,
    )
    return _replace_gate_root_zs(poses, root_zs_m)


def _extend_gate_poses(
    template_poses: list[tuple[float, float, float, float, float, float]],
    *,
    gate_count: int,
    gate_spacing_m: int,
    randomize_gate_height: bool,
    sequential_gate_height_step_m: float | None,
    rng: random.Random,
) -> list[tuple[float, float, float, float, float, float]]:
    spacing_m = float(gate_spacing_m)
    poses = list(template_poses[:gate_count])

    while len(poses) < gate_count:
        prev_pose = poses[-1] if poses else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        prev_x_m, prev_y_m = prev_pose[0], prev_pose[1]
        x_m = prev_x_m + spacing_m
        y_limit_m = min(4.0, 1.2 + 0.45 * len(poses))
        y_m = max(-y_limit_m, min(y_limit_m, prev_y_m))
        yaw_rad = math.atan2(y_m - prev_y_m, x_m - prev_x_m)
        poses.append((x_m, y_m, 0.0, 0.0, 0.0, yaw_rad))

    root_zs_m = _random_gate_root_zs(
        rng,
        gate_count=gate_count,
        randomize_gate_height=randomize_gate_height,
        sequential_gate_height_step_m=sequential_gate_height_step_m,
    )
    return _replace_gate_root_zs(poses, root_zs_m)


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


def _existing_gate_models(world: ET.Element) -> dict[int, ET.Element]:
    gates: dict[int, ET.Element] = {}
    for model in world.findall("./model"):
        name = model.get("name", "")
        prefix = "racing_gate_"
        if not name.startswith(prefix):
            continue
        try:
            gate_idx = int(name[len(prefix):])
        except ValueError:
            continue
        gates[gate_idx] = model
    return gates


def _configure_gate_models(world: ET.Element, gate_count: int) -> None:
    existing = _existing_gate_models(world)
    if not existing:
        raise RuntimeError(f"No racing_gate_* models found in {SRC}.")

    template = copy.deepcopy(existing[min(existing)])
    for gate_idx, model in list(existing.items()):
        if gate_idx > gate_count:
            world.remove(model)

    existing = _existing_gate_models(world)
    for gate_idx in range(1, gate_count + 1):
        if gate_idx in existing:
            existing[gate_idx].set("name", f"racing_gate_{gate_idx}")
            continue
        model = copy.deepcopy(template)
        model.set("name", f"racing_gate_{gate_idx}")
        world.append(model)


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


def _find_toml_section(lines: list[str], section_name: str) -> tuple[int, int]:
    header = f"[{section_name}]"
    start_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == header:
            start_idx = idx
            break
    if start_idx < 0:
        raise RuntimeError(f"No [{section_name}] section found in runtime config.")

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end_idx = idx
            break
    return start_idx, end_idx


def _find_toml_key(lines: list[str], start_idx: int, end_idx: int, key: str) -> int:
    for idx in range(start_idx + 1, end_idx):
        stripped = lines[idx].lstrip()
        if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
            return idx
    return -1


def _find_toml_array_end(lines: list[str], start_idx: int) -> int:
    depth = 0
    saw_array = False
    for idx in range(start_idx, len(lines)):
        line = lines[idx].split("#", 1)[0]
        depth += line.count("[") - line.count("]")
        saw_array = saw_array or "[" in line
        if saw_array and depth <= 0:
            return idx
    raise RuntimeError(f"Could not find TOML array end starting at line {start_idx + 1}.")


def _format_known_gate_positions(gates_neu: list[tuple[float, float, float]]) -> list[str]:
    lines = ["known_gate_positions_neu = [\n"]
    for gate in gates_neu:
        lines.append(f"  [{gate[0]:.6f}, {gate[1]:.6f}, {gate[2]:.6f}],\n")
    lines.append("]\n")
    return lines


def _replace_toml_array(
    lines: list[str],
    section_name: str,
    key: str,
    replacement: list[str],
) -> list[str]:
    section_start, section_end = _find_toml_section(lines, section_name)
    key_idx = _find_toml_key(lines, section_start, section_end, key)
    if key_idx < 0:
        insert_idx = section_end
        return lines[:insert_idx] + replacement + lines[insert_idx:]

    array_end_idx = _find_toml_array_end(lines, key_idx)
    return lines[:key_idx] + replacement + lines[array_end_idx + 1 :]


def _replace_toml_scalar(
    lines: list[str],
    section_name: str,
    key: str,
    value: str,
) -> list[str]:
    section_start, section_end = _find_toml_section(lines, section_name)
    key_idx = _find_toml_key(lines, section_start, section_end, key)
    replacement = f"{key} = {value}\n"
    if key_idx < 0:
        return lines[:section_end] + [replacement] + lines[section_end:]
    return lines[:key_idx] + [replacement] + lines[key_idx + 1 :]


def _update_runtime_config(
    path: Path,
    *,
    gates_neu: list[tuple[float, float, float]],
    gate_count: int,
) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    gate_lines = _format_known_gate_positions(gates_neu)

    # These two sections are sim/debug-only consumers of known gate coordinates.
    # Do not update [state_estimation].known_gate_positions_neu here, because that
    # can accidentally make estimator tests depend on known sim gates.
    lines = _replace_toml_array(
        lines,
        "perception_geometry_audit",
        "known_gate_positions_neu",
        gate_lines,
    )
    lines = _replace_toml_array(
        lines,
        "gate_source",
        "known_gate_positions_neu",
        gate_lines,
    )
    lines = _replace_toml_scalar(lines, "race", "gate_count", str(int(gate_count)))

    updated = "".join(lines)
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _set_text(parent: ET.Element, path: str, text: str) -> None:
    element = parent.find(path)
    if element is None:
        raise RuntimeError(f"Missing required SDF element: {path}")
    element.text = text


def _rgba(r: float, g: float, b: float, a: float = 1.0) -> str:
    return f"{r:.6f} {g:.6f} {b:.6f} {a:.6f}"


def _apply_gate_material(
    world: ET.Element,
    *,
    gate_count: int,
    rgb: tuple[float, float, float],
    specular: float,
) -> None:
    for gate_idx in range(1, gate_count + 1):
        model = _find_gate_model(world, gate_idx)
        for material in model.findall(".//material"):
            ambient = material.find("ambient")
            diffuse = material.find("diffuse")
            spec = material.find("specular")
            if ambient is not None:
                ambient.text = _rgba(rgb[0], rgb[1], rgb[2])
            if diffuse is not None:
                diffuse.text = _rgba(rgb[0], rgb[1], rgb[2])
            if spec is not None:
                spec.text = _rgba(specular, specular, specular)


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


def _randomize_gate_material(
    world: ET.Element,
    rng: random.Random,
    gate_count: int,
) -> dict[str, float]:
    blue = max(0.35, min(1.0, 0.701961 * rng.uniform(0.75, 1.15)))
    specular = rng.uniform(0.03, 0.15)
    _apply_gate_material(
        world,
        gate_count=gate_count,
        rgb=(0.0, 0.0, blue),
        specular=specular,
    )

    return {"blue": blue, "specular": specular}


def _randomize_gate_color(
    world: ET.Element,
    rng: random.Random,
    gate_count: int,
) -> dict[str, float]:
    hue = rng.random()
    saturation = rng.uniform(0.65, 0.95)
    value = rng.uniform(0.65, 1.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    specular = rng.uniform(0.03, 0.18)
    _apply_gate_material(
        world,
        gate_count=gate_count,
        rgb=(red, green, blue),
        specular=specular,
    )

    return {
        "hue": hue,
        "saturation": saturation,
        "value": value,
        "red": red,
        "green": green,
        "blue": blue,
        "specular": specular,
    }


def _set_gate_rgb(
    world: ET.Element,
    *,
    gate_count: int,
    rgb: tuple[float, float, float],
) -> dict[str, float]:
    red, green, blue = rgb
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    _apply_gate_material(
        world,
        gate_count=gate_count,
        rgb=rgb,
        specular=FIXED_GATE_RGB_SPECULAR,
    )

    return {
        "hue": hue,
        "saturation": saturation,
        "value": value,
        "red": red,
        "green": green,
        "blue": blue,
        "specular": FIXED_GATE_RGB_SPECULAR,
    }


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
    template_poses = [
        _get_gate_pose(world, gate_idx)
        for gate_idx in range(
            1,
            min(DEFAULT_GATE_COUNT, options.gate_count) + 1,
        )
    ]
    _configure_gate_models(world, options.gate_count)

    if options.randomize_positions:
        poses = _generate_gate_poses(
            rng,
            gate_count=options.gate_count,
            gate_spacing_m=options.gate_spacing_m,
            randomize_gate_height=options.randomize_gate_height,
            sequential_gate_height_step_m=options.sequential_gate_height_step_m,
        )
    else:
        poses = _extend_gate_poses(
            template_poses,
            gate_count=options.gate_count,
            gate_spacing_m=options.gate_spacing_m,
            randomize_gate_height=options.randomize_gate_height,
            sequential_gate_height_step_m=options.sequential_gate_height_step_m,
            rng=rng,
        )

    for gate_idx, pose_values in enumerate(poses, start=1):
        _set_gate_pose(world, gate_idx, pose_values)

    lighting = _randomize_lighting(world, rng) if options.randomize_lighting else None
    if options.gate_rgb is not None:
        gate_color = _set_gate_rgb(
            world,
            gate_count=options.gate_count,
            rgb=options.gate_rgb,
        )
    else:
        gate_color = (
            _randomize_gate_color(world, rng, options.gate_count)
            if options.randomize_gate_color
            else None
        )
    gate_material = None
    if gate_color is None and options.randomize_gate_material:
        gate_material = _randomize_gate_material(world, rng, options.gate_count)
    distractor_count = _add_distractors(world, rng) if options.add_distractors else 0
    obstacle_count = _add_obstacles(world, rng, poses) if options.add_obstacles else 0

    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="UTF-8", xml_declaration=True)
    return WorldRandomizationResult(
        poses=poses,
        lighting=lighting,
        gate_material=gate_material,
        gate_color=gate_color,
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
    parser.add_argument(
        "--gate-count",
        type=int,
        default=DEFAULT_GATE_COUNT,
        help=(
            "Number of racing gates to place in the generated world. "
            f"Default: {DEFAULT_GATE_COUNT}."
        ),
    )
    parser.add_argument(
        "--gate-spacing-m",
        type=int,
        default=DEFAULT_GATE_SPACING_M,
        help=(
            "Approximate forward/course spacing between gates in meters. "
            "This maps to SDF x and pilot forward/y. "
            f"Default: {DEFAULT_GATE_SPACING_M}."
        ),
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=RUNTIME_CONFIG,
        help=f"Runtime TOML to update with generated gate centers. Default: {RUNTIME_CONFIG}.",
    )
    _add_bool_arg(
        parser,
        "--update-runtime",
        default=True,
        help=(
            "Update [gate_source]/[perception_geometry_audit] known_gate_positions_neu "
            "and [race].gate_count after generating the world."
        ),
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
    parser.add_argument(
        "--sequential-gate-height-step-m",
        "--gate-height-step-m",
        type=float,
        dest="sequential_gate_height_step_m",
        default=None,
        help=(
            "Enable sequential gate root-z randomization. Gate 1 is sampled in "
            f"[0, {SEQUENTIAL_FIRST_GATE_ROOT_Z_MAX_M:.1f}] m; each later gate "
            "is sampled within +/- this many meters of the previous gate, "
            "clamped at z >= 0. This overrides --randomize-gate-height."
        ),
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
    _add_bool_arg(
        parser,
        "--randomize-gate-color",
        default=False,
        help=(
            "Apply one stronger whole-track gate RGB color augmentation. "
            "This takes precedence over --randomize-gate-material."
        ),
    )
    parser.add_argument(
        "--gate-rgb",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help=(
            "Set all gate materials to a fixed normalized RGB color, with each "
            "component in [0.0, 1.0]. This overrides --randomize-gate-color "
            "and --randomize-gate-material."
        ),
    )
    args = parser.parse_args()
    if args.gate_count < 1:
        parser.error("--gate-count must be at least 1.")
    if args.gate_spacing_m < 1:
        parser.error("--gate-spacing-m must be at least 1.")
    if (
        args.sequential_gate_height_step_m is not None
        and args.sequential_gate_height_step_m <= 0.0
    ):
        parser.error("--sequential-gate-height-step-m must be greater than 0.")
    gate_rgb: tuple[float, float, float] | None = None
    if args.gate_rgb is not None:
        red, green, blue = (float(value) for value in args.gate_rgb)
        gate_rgb = (red, green, blue)
        if any(value < 0.0 or value > 1.0 for value in gate_rgb):
            parser.error(
                "--gate-rgb values must be normalized floats in [0.0, 1.0]."
            )

    seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    options = RandomizationOptions(
        gate_count=int(args.gate_count),
        gate_spacing_m=int(args.gate_spacing_m),
        randomize_positions=bool(args.randomize_positions),
        randomize_gate_height=bool(args.randomize_gate_height),
        sequential_gate_height_step_m=(
            None
            if args.sequential_gate_height_step_m is None
            else float(args.sequential_gate_height_step_m)
        ),
        randomize_lighting=bool(args.randomize_lighting),
        add_distractors=bool(args.add_distractors),
        add_obstacles=bool(args.add_obstacles),
        randomize_gate_material=bool(args.randomize_gate_material),
        randomize_gate_color=bool(args.randomize_gate_color),
        gate_rgb=gate_rgb,
    )
    result = _write_random_world(seed, options)
    gate_centers_neu = [
        _pilot_neu_center_from_sdf_pose(pose_values)
        for pose_values in result.poses
    ]
    runtime_updated = False
    if bool(args.update_runtime):
        runtime_updated = _update_runtime_config(
            Path(args.runtime_config),
            gates_neu=gate_centers_neu,
            gate_count=options.gate_count,
        )

    print(f"seed={seed}")
    print(f"template={SRC}")
    print(f"wrote={OUTPUT}")
    if bool(args.update_runtime):
        print(
            f"runtime_config={Path(args.runtime_config)} "
            f"updated={int(runtime_updated)} "
            "sections=perception_geometry_audit,gate_source,race"
        )
    else:
        print("runtime_config_update=0")
    print(f"launch=PX4_GZ_WORLD={OUTPUT_WORLD_NAME} make px4_sitl gz_racer_mono_cam")
    print(
        "options="
        f"gate_count={options.gate_count} "
        f"gate_spacing_m={options.gate_spacing_m} "
        f"positions={int(options.randomize_positions)} "
        f"gate_height={int(options.randomize_gate_height)} "
        f"sequential_gate_height_step_m={options.sequential_gate_height_step_m} "
        f"lighting={int(options.randomize_lighting)} "
        f"distractors={int(options.add_distractors)} "
        f"obstacles={int(options.add_obstacles)} "
        f"gate_material={int(options.randomize_gate_material)} "
        f"gate_color={int(options.randomize_gate_color)} "
        f"gate_rgb={options.gate_rgb}"
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
    if result.gate_color is not None:
        print(
            "gate_color "
            f"rgb=({result.gate_color['red']:.3f},"
            f"{result.gate_color['green']:.3f},"
            f"{result.gate_color['blue']:.3f}) "
            f"hsv=({result.gate_color['hue']:.3f},"
            f"{result.gate_color['saturation']:.3f},"
            f"{result.gate_color['value']:.3f}) "
            f"specular={result.gate_color['specular']:.3f}"
        )
    if result.distractor_count:
        print(f"distractors={result.distractor_count}")
    if result.obstacle_count:
        print(f"obstacles={result.obstacle_count}")
    for gate_idx, pose_values in enumerate(result.poses, start=1):
        pilot_neu = gate_centers_neu[gate_idx - 1]
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
