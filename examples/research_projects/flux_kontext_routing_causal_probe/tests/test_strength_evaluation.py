from pathlib import Path

from strength_overfit_evaluation import PerceptualModels


def test_hub_cache_snapshot_resolution(tmp_path: Path):
    root = tmp_path / "models--openai--clip"
    snapshot = root / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    assert PerceptualModels._resolve_local_model_dir(str(root)) == str(snapshot)


def test_direct_model_directory_resolution(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    assert PerceptualModels._resolve_local_model_dir(str(tmp_path)) == str(tmp_path)
