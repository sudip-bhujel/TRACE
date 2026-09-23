"""CPU check: python -m attacker.tools.test_plot_aggregation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from attacker.tools.plot_aggregation import load_exports, plot_sequence, plot_summary


def test_plot_aggregation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        record = {"sequence_index": 0, "target": [[0.25, 0.75, 0, 0, 0]] * 8,
                  "prediction": [[0.2] * 5] * 8}
        payload = dict(h5_path="test.h5", sequence_length=8, stride=8, batch_size=1,
                       aggregation_size=4, teacher_forcing=False,
                       action_names=["MoveAhead", "RotateLeft", "RotateRight", "LookDown", "LookUp"],
                       sequences=[record])
        rgb = np.full((8, 3, 8, 8), 0.5, dtype=np.float32)
        for side in ("before", "after"):
            (root / side).mkdir()
            (root / side / "action_histograms.json").write_text(json.dumps(payload))
            np.savez_compressed(root / side / "reconstruction_seq_001.npz",
                                target_images=rgb, predicted_images=rgb)
        names, before, after = load_exports(root)
        close = plt.close
        def check_layout(fig):
            if hasattr(fig, "axes"):
                assert fig._suptitle is None
                assert fig.axes[0].title.get_fontfamily() == ["DejaVu Serif"]
                assert [ax.get_title().split("\n")[0] for ax in fig.axes[:8]] == [
                    f"Window {i + 1}" for i in range(8)]
                if len(fig.axes) == 24:
                    assert all(fig.axes[i].texts[0].get_rotation() == 90 for i in (0, 8, 16))
            if hasattr(fig, "axes") and len(fig.axes) == 8:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                titles = [ax.title.get_window_extent(renderer) for ax in fig.axes]
                for offset in (0, 4):
                    assert all(titles[offset + i].x1 < titles[offset + i + 1].x0 for i in range(3))
                for ax in fig.axes:
                    ticks = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
                    assert all(a.x1 < b.x0 for a, b in zip(ticks, ticks[1:]))
            return close(fig)
        with patch("matplotlib.pyplot.close", side_effect=check_layout):
            plot_sequence(root, root / "figures", 1, names, before[0], after[0])
        assert len(list((root / "figures").glob("*"))) == 4
        target = before[0][0]
        alpha = np.linspace(0, 1, 8)[:, None]
        ranked_after = {0: [target, (1 - alpha) * target + alpha * np.array([1, 0, 0, 0, 0])]}
        plot_summary(root, root / "summary", names, before, ranked_after)
        summary = json.loads((root / "summary/summary.json").read_text())
        assert len(list((root / "summary").glob("*"))) == 7
        assert summary["num_windows"] == 8
        assert [x["window_number"] for x in summary["examples"]] == [2, 5, 7]
        assert np.isclose(summary["mean_tv"]["After fine-tuning"], 0.375)
        assert np.isclose(summary["mean_tv"]["Uniform baseline"], 0.6)
        scratch_path = root / "scratch.json"
        payload["sequences"][0]["prediction"] = payload["sequences"][0]["target"]
        scratch_path.write_text(json.dumps(payload))
        plot_summary(root, root / "summary", names, before, ranked_after, scratch_path=scratch_path)
        summary = json.loads((root / "summary/summary.json").read_text())
        assert summary["mean_tv"]["From scratch"] == 0
        assert summary["scratch_export"] == str(scratch_path.resolve())
        agg8_path = root / "agg8.json"
        agg8_payload = {**payload, "h5_path": "agg8.h5", "aggregation_size": 8,
                        "sequences": [
                            {"sequence_index": 0, "target": [[0.125, 0.875, 0, 0, 0]] * 8,
                             "prediction": [[0.2] * 5] * 8},
                            {"sequence_index": 1, "target": [[0.5, 0.5, 0, 0, 0]] * 8,
                             "prediction": [[0.5, 0.5, 0, 0, 0]] * 8}]}
        agg8_path.write_text(json.dumps(agg8_payload))
        panel_colors = []
        def check_panels(fig):
            if hasattr(fig, "axes") and len(fig.axes) == 1:
                ax = fig.axes[0]
                assert fig._suptitle is None and ax.get_title() == ax.get_xlabel() == ""
                assert np.allclose(np.diff(ax.get_xticks()), 0.7)
                assert ax.get_ylim() == (0, 1)
                assert all(text.get_text().startswith(r"$\mu=") for text in ax.texts)
                fig.canvas.draw()
                ticks = [label.get_window_extent(fig.canvas.get_renderer()) for label in ax.get_xticklabels()]
                assert all(a.x1 < b.x0 for a, b in zip(ticks, ticks[1:]))
                panel_colors.append([box.get_facecolor() for box in ax.patches])
            return close(fig)
        with patch("matplotlib.pyplot.close", side_effect=check_panels):
            plot_summary(root, root / "summary", names, before, ranked_after,
                         scratch_path=scratch_path, agg8_path=agg8_path)
        assert [len(colors) for colors in panel_colors] == [4, 2]
        assert panel_colors[0][2:] == panel_colors[1]
        assert len(list((root / "summary").glob("*"))) == 9
        assert (root / "summary/action_error_summary_agg8.pdf").is_file()
        summary = json.loads((root / "summary/summary.json").read_text())
        assert summary["agg8"]["num_sequences"] == 2 and summary["agg8"]["num_windows"] == 16
        assert np.isclose(summary["agg8"]["mean_tv"]["From scratch"], 0.3375)
        assert np.isclose(summary["agg8"]["mean_tv"]["Uniform baseline"], 0.6375)
        assert np.isclose(summary["mean_tv"]["Uniform baseline"], 0.6)
        agg8_payload["aggregation_size"] = 4
        agg8_path.write_text(json.dumps(agg8_payload))
        try:
            plot_summary(root, root / "summary", names, before, ranked_after, agg8_path=agg8_path)
        except ValueError as exc:
            assert "requires Agg4 and Agg8" in str(exc)
        else:
            raise AssertionError("Wrong aggregation size was accepted")
        payload["stride"] = 2
        scratch_path.write_text(json.dumps(payload))
        try:
            load_exports(root, scratch_path)
        except ValueError as exc:
            assert "metadata differs: stride" in str(exc)
        else:
            raise AssertionError("Mismatched scratch metadata was accepted")
        payload["stride"] = 8
        payload["sequences"][0]["target"] = [[0.75, 0.25, 0, 0, 0]] * 8
        (root / "after/action_histograms.json").write_text(json.dumps(payload))
        try:
            load_exports(root)
        except ValueError as exc:
            assert "Ground-truth actions differ" in str(exc)
        else:
            raise AssertionError("Mismatched targets were accepted")
        np.savez_compressed(root / "after/reconstruction_seq_001.npz",
                            target_images=rgb * 0, predicted_images=rgb)
        try:
            plot_sequence(root, root / "figures", 1, names, before[0], after[0])
        except ValueError as exc:
            assert "Ground-truth images differ" in str(exc)
        else:
            raise AssertionError("Mismatched images were accepted")
    print("Histogram/image plotting and before/after alignment checks passed.")


if __name__ == "__main__":
    test_plot_aggregation()
