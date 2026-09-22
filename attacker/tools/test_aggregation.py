"""CPU check: python -m attacker.tools.test_aggregation. No pretrained downloads."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from attacker.data.dataset import TemporalGradientDataset, episode_split
from attacker.evaluation.evaluate import evaluate_and_save_reconstructions
from attacker.evaluation.loss import TemporalCombinedLoss
from attacker.evaluation.metrics import action_metric
from attacker.tools.aggregate_gradients import aggregate_file
from attacker.training.train import train


class SmallAttacker(nn.Module):
    """Exercise the existing training entry point without allocating the 2B model."""

    def __init__(self, **kwargs):
        super().__init__()
        self.head = nn.Linear(4, 5)
        self.image = nn.Parameter(torch.zeros(3, 16, 16))

    def forward(self, gradients, **kwargs):
        logits = self.head(gradients)
        images = self.image.sigmoid().expand(*gradients.shape[:2], 3, 16, 16)
        return images, logits, gradients, None


class PixelDistance(nn.Module):
    def forward(self, pred, target):
        return (pred - target).square().mean((1, 2, 3), keepdim=True)


def test_aggregation(group_size=4):
    torch.set_num_threads(1)
    scale = group_size // 4
    n = 42 * scale
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "raw.h5", root / "agg.h5"
        gradients = np.arange(n * 4, dtype=np.float16).reshape(n, 4) / 100
        images = np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None, None], (n, 3, 16, 16))
        actions = np.arange(n, dtype=np.int64) % 5
        episodes = np.repeat([10, 11, 12], 14 * scale)
        done = np.zeros(n, dtype=bool)
        done[np.array([6, 14, 28, 42]) * scale - 1] = True  # A terminal inside the first episode ID.
        with h5py.File(source, "w") as f:
            for key, value in dict(gradients=gradients, images=images, actions=actions,
                                   episode_ids=episodes, done=done).items():
                f[key] = value
            f.attrs.update(num_actions=5, completed_steps=n, loss_type="ppo")
        aggregate_file(source, output, group_size=group_size)
        with h5py.File(output, "r") as f:
            starts = [start * scale for start in [0, 6, 10, 14, 18, 22, 28, 32, 36]]
            assert f["source_start"][:].tolist() == starts
            assert f.attrs["aggregation_complete"] and f.attrs["dropped_steps"] == 6 * scale
            assert f.attrs["aggregation_size"] == group_size
            for i, start in enumerate(starts):
                np.testing.assert_allclose(f["gradients"][i], gradients[start:start + group_size].astype("f4").mean(0))
                np.testing.assert_array_equal(f["images"][i], images[start + group_size - 1])
                np.testing.assert_allclose(f["actions"][i], np.bincount(actions[start:start + group_size], minlength=5) / group_size)
                assert not done[start:start + group_size - 1].any()
        try:
            aggregate_file(source, output)
            raise AssertionError("Existing files must not be overwritten")
        except FileExistsError:
            pass

        dataset = TemporalGradientDataset(str(output), sequence_length=2, stride=1)
        train_set, val_set = episode_split(dataset)
        def groups(subset):
            return {int(dataset.split_episode_ids[dataset.sequence_indices[i][0]]) for i in subset.indices}
        assert groups(train_set).isdisjoint(groups(val_set))
        assert train_set.indices == episode_split(dataset)[0].indices
        assert dataset[0][2].shape == (2, 5)
        assert dataset[0][2].dtype == torch.float32

        # Confident wrong classifiers must still receive histogram gradients.
        logits = torch.tensor([[[20., -20., -20., -20., -20.]]], requires_grad=True)
        target = torch.full_like(logits, 0.2)
        rgb = torch.zeros(1, 1, 3, 16, 16)
        criterion = TemporalCombinedLoss(action_weight=1, lpips_weight=0)
        loss, _ = criterion(rgb, rgb, logits, target)
        loss.backward()
        assert logits.grad.abs().sum() > 0
        key, tv = action_metric(torch.zeros_like(logits), target)
        assert key == "action_histogram_tv" and torch.allclose(tv, torch.zeros_like(tv))
        _, wrong_tv = action_metric(logits, target)
        assert torch.allclose(wrong_tv, torch.tensor([[0.8]]))

        # Exercise four-step fine-tuning and eight-step training from scratch.
        pretrained = root / "base.pt"
        torch.save({"model_state_dict": SmallAttacker().state_dict(), "epoch": 50}, pretrained)
        with patch("attacker.training.train.AutoregressiveGradientInversion", SmallAttacker):
            train(h5_path=str(output), save_dir=root / "trained", device="cpu",
                  model_type="autoregressive",
                  pretrained_checkpoint=str(pretrained) if group_size == 4 else None,
                  num_epochs=1, batch_size=2, num_workers=0, sequence_length=2,
                  stride=1, split_by_episode=True, num_actions=5, lpips_weight=0)
        checkpoint = torch.load(root / "trained/best_model.pt", weights_only=False)
        assert checkpoint["epoch"] == 1 and checkpoint["aggregation_size"] == group_size
        assert "action_histogram_tv" in checkpoint["val_loss"]
        model = SmallAttacker()
        model.load_state_dict(checkpoint["model_state_dict"])
        with patch("attacker.evaluation.metrics.lpips.LPIPS", return_value=PixelDistance()):
            evaluate_and_save_reconstructions(model, DataLoader(dataset, batch_size=1),
                torch.device("cpu"), root / "eval", num_sequences=1,
                enable_fid=False, model_type="autoregressive")
        metrics = json.loads((root / "eval/metrics.json").read_text())
        assert "action_histogram_tv" in metrics and "action_accuracy" not in metrics
        assert not (root / "eval/confusion_matrix.png").exists()

        # Export-only evaluation must not load LPIPS/FID or replace old metrics.
        export_dir = root / "export"
        export_dir.mkdir()
        (export_dir / "metrics.json").write_text("existing metrics")
        with patch("attacker.evaluation.evaluate.MetricsComputer", side_effect=AssertionError("Metrics should be skipped")):
            evaluate_and_save_reconstructions(model, DataLoader(dataset, batch_size=1),
                torch.device("cpu"), export_dir, num_sequences=2,
                model_type="autoregressive", compute_metrics=False,
                checkpoint_path=str(pretrained))
        exported = json.loads((export_dir / "action_histograms.json").read_text())
        assert exported["checkpoint"] == str(pretrained)
        assert exported["h5_path"] == str(output)
        assert len(exported["sequences"]) == 2
        for i, record in enumerate(exported["sequences"]):
            g, target_rgb, target = dataset[i]
            with torch.no_grad():
                predicted_rgb, export_logits, _, _ = model(g.unsqueeze(0))
            assert record["sequence_index"] == i
            np.testing.assert_allclose(record["target"], target.numpy())
            np.testing.assert_allclose(record["prediction"], export_logits[0].softmax(-1).numpy())
            _, tv = action_metric(export_logits, target.unsqueeze(0))
            np.testing.assert_allclose(record["tv"], tv[0].numpy())
            with np.load(export_dir / f"reconstruction_seq_{i + 1:03d}.npz") as saved:
                np.testing.assert_allclose(saved["target_images"], target_rgb.numpy())
                np.testing.assert_allclose(saved["predicted_images"], predicted_rgb[0].numpy())
        assert (export_dir / "metrics.json").read_text() == "existing metrics"

        # The original scalar-label path still computes ordinary accuracy/loss.
        raw = TemporalGradientDataset(str(source), sequence_length=2)
        assert raw.action_target == "class" and raw[0][2].dtype == torch.long
        key, accuracy = action_metric(logits.detach(), torch.zeros((1, 1), dtype=torch.long))
        assert key == "accuracy" and accuracy.item() == 100
        loss, _ = criterion(rgb, rgb, logits, torch.ones((1, 1), dtype=torch.long))
        assert loss.item() == 2.0  # Existing scalar-label clipping is unchanged.
        dataset.__del__()
        raw.__del__()
    print(f"{group_size}-step aggregation, episode split, training, evaluation, and scalar-label checks passed.")


if __name__ == "__main__":
    test_aggregation()
    test_aggregation(group_size=8)
