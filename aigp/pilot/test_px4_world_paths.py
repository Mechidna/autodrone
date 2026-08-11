from autonomy_wrapper import PyAIPilotAutonomyAPI
from autonomy_core.tools.px4_gazebo_paths import PX4_WORLDS_DIR_ENV


def test_runtime_world_candidates_honor_portable_environment(tmp_path, monkeypatch):
    worlds_dir = tmp_path / "worlds"
    worlds_dir.mkdir()
    expected = worlds_dir / "portable_world.sdf"
    monkeypatch.setenv(PX4_WORLDS_DIR_ENV, str(worlds_dir))

    candidates = PyAIPilotAutonomyAPI._world_sdf_candidates("portable_world")

    assert expected.resolve() in candidates


def test_runtime_world_candidates_preserve_explicit_sdf(tmp_path):
    explicit = tmp_path / "explicit.sdf"

    candidates = PyAIPilotAutonomyAPI._world_sdf_candidates(str(explicit))

    assert explicit in candidates
