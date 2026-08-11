from argparse import Namespace
import xml.etree.ElementTree as ET

import pytest

import randomize_gate_world as randomizer


def _args(**overrides):
    values = {
        "source_world_sdf": None,
        "output_world_sdf": None,
        "output_world_name": None,
        "px4_root": None,
        "worlds_dir": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _write_minimal_source(path):
    models = "".join(
        f'<model name="racing_gate_{index}"><pose>{index * 8} 0 0 0 0 0</pose></model>'
        for index in range(1, 4)
    )
    path.write_text(
        f'<?xml version="1.0"?><sdf version="1.9"><world name="source">{models}</world></sdf>',
        encoding="utf-8",
    )


def test_resolve_world_paths_accepts_explicit_portable_files(tmp_path):
    source = tmp_path / "source.sdf"
    output = tmp_path / "generated.sdf"
    _write_minimal_source(source)

    resolved = randomizer._resolve_world_paths(
        _args(source_world_sdf=source, output_world_sdf=output)
    )

    assert resolved.source_sdf == source.resolve()
    assert resolved.output_sdf == output.resolve()
    assert resolved.output_world_name == "generated"
    assert resolved.px4_root is None


def test_resolve_world_paths_uses_explicit_px4_root(tmp_path):
    px4_root = tmp_path / "PX4-Autopilot"
    worlds_dir = px4_root / randomizer.PX4_WORLDS_RELATIVE
    worlds_dir.mkdir(parents=True)
    source = worlds_dir / f"{randomizer.DEFAULT_SOURCE_WORLD_NAME}.sdf"
    _write_minimal_source(source)

    resolved = randomizer._resolve_world_paths(_args(px4_root=px4_root))

    assert resolved.source_sdf == source.resolve()
    assert resolved.output_sdf == (
        worlds_dir / f"{randomizer.DEFAULT_OUTPUT_WORLD_NAME}.sdf"
    ).resolve()
    assert resolved.px4_root == px4_root.resolve()


def test_resolve_world_paths_rejects_name_filename_mismatch(tmp_path):
    source = tmp_path / "source.sdf"
    _write_minimal_source(source)

    with pytest.raises(ValueError, match="must match"):
        randomizer._resolve_world_paths(
            _args(
                source_world_sdf=source,
                output_world_sdf=tmp_path / "generated.sdf",
                output_world_name="different",
            )
        )


def test_resolve_world_paths_requires_sdf_output_suffix(tmp_path):
    source = tmp_path / "source.sdf"
    _write_minimal_source(source)

    with pytest.raises(ValueError, match="end in .sdf"):
        randomizer._resolve_world_paths(
            _args(
                source_world_sdf=source,
                output_world_sdf=tmp_path / "generated.xml",
            )
        )


def test_write_random_world_uses_resolved_paths_and_world_name(tmp_path):
    source = tmp_path / "source.sdf"
    output = tmp_path / "portable_world.sdf"
    _write_minimal_source(source)

    result = randomizer._write_random_world(
        123,
        randomizer.RandomizationOptions(),
        source_sdf=source,
        output_sdf=output,
        output_world_name="portable_world",
    )

    assert len(result.poses) == randomizer.DEFAULT_GATE_COUNT
    root = ET.parse(output).getroot()
    assert root.find("world").get("name") == "portable_world"
