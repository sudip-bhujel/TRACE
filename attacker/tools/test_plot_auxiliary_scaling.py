"""Run with python -m attacker.tools.test_plot_auxiliary_scaling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt

from attacker.tools.plot_auxiliary_scaling import FRACTIONS, METRICS, load_metrics, plot_metrics


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for fraction in FRACTIONS:
            path = root / str(fraction) / "eval_results/metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"num_images": 800,
                                       **{m: fraction / 100 for m in METRICS}}))
        values = load_metrics(root)
        close = plt.close
        checked = []

        def check_figure(fig):
            if not hasattr(fig, "axes"):
                return close(fig)
            line, = fig.axes[0].lines
            assert list(line.get_xdata()) == list(FRACTIONS)
            assert list(line.get_ydata()) == [f / 100 for f in FRACTIONS]
            assert len(fig.axes) == 1
            checked.append(fig.axes[0].get_ylabel())
            close(fig)

        with patch.object(plt, "close", side_effect=check_figure):
            plot_metrics(values, root / "figures")
        assert checked == list(METRICS.values())
        for metric in METRICS:
            assert (root / f"figures/auxiliary_scaling_{metric}.pdf").stat().st_size > 0
    print("All four plots preserve the saved metric values and data fractions.")


if __name__ == "__main__":
    main()
