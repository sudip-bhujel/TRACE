from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from attacker.tools import extract_wandb_results
from attacker.tools.copy_remote_metrics import metric_sources
from attacker.tools.extract_wandb_results import parse_output_log
from attacker.tools.export_wandb import _config_paths, _latest_finished_run, _safe_parts


class FakeApi:
    def runs(self, project_path, filters):
        assert project_path == "entity/project"
        assert filters == {"display_name": "run", "state": "finished", "group": "group"}
        return [
            SimpleNamespace(created_at="2026-01-01", id="old"),
            SimpleNamespace(created_at="2026-02-01", id="new"),
        ]


def test_latest_finished_run():
    assert _latest_finished_run(FakeApi(), "entity/project", "run", "group").id == "new"


def test_config_directory_and_hierarchy():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "nested").mkdir()
        (root / "one.yaml").touch()
        (root / "nested" / "two.yml").touch()
        assert _config_paths([directory]) == [root / "nested" / "two.yml", root / "one.yaml"]
    assert _safe_parts("ppo/architecture actions") == ("ppo", "architecture_actions")


def test_result_keys_preserve_groups():
    a2c = {"run": {"group": "a2c/ablation", "name": "ablation-t8"}}
    ppo = {"run": {"group": "ppo/ablation", "name": "ablation-t8"}}
    assert extract_wandb_results._result_key(a2c) != extract_wandb_results._result_key(ppo)


def test_parse_output_log():
    result = parse_output_log(
        """Using device: cuda:0
DDP enabled: world_size=6, accumulation_steps=2
Flash Attention available: True
Loading data from train.h5 (lazy loading for gradients)...
  Gradient dim limited to 936,102
Dataset initialized: 49,701 sequences, seq_len=8, stride=2
Dataset split:
  Train: 47,215 sequences
  Val: 2,486 sequences
  Number of actions: 5
Model parameters: 2,239,105,364
Training complete! Best val loss: 0.0228
Using device: cuda
Loading data from test.h5 (lazy loading for gradients)...
Dataset initialized: 5,541 sequences, seq_len=8, stride=2
Evaluation Summary (800 images from 100 sequences)
  MSE: 0.014619 +/- 0.005578
  PSNR: 18.62 +/- 1.50 dB
  SSIM: 0.5654 +/- 0.0892
  MS-SSIM: 0.7192 +/- 0.0696
  LPIPS: 0.3899 +/- 0.0722
  FID: 289.34
  Action Accuracy: 97.5 %
  Per-timestep metrics:
  t   PSNR        SSIM        LPIPS
  0   18.07       0.5526      0.4040
"""
    )
    assert result["dataset"]["training"]["available_sequences"] == 49701
    assert result["model"]["attacker_parameters"] == 2239105364
    assert result["evaluation"]["metrics"]["psnr"]["mean"] == 18.62
    assert result["evaluation"]["per_timestep"][0]["timestep"] == 0


def test_metric_sources():
    with TemporaryDirectory() as directory:
        config_dir = Path(directory)
        (config_dir / "run.yaml").write_text(
            """eval:
  enabled: true
output:
  save_dir: ckpts/trace/ppo/example
wandb:
  name: example-run
"""
        )
        assert metric_sources(config_dir, "/remote/project") == {
            "/remote/project/ckpts/trace/ppo/example/eval_results/metrics.json": "example-run"
        }


if __name__ == "__main__":
    test_latest_finished_run()
    test_config_directory_and_hierarchy()
    test_result_keys_preserve_groups()
    test_parse_output_log()
    test_metric_sources()
    print("ok")
