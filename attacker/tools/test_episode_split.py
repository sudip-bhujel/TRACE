"""Run with python -m attacker.tools.test_episode_split; no data files needed."""

from types import SimpleNamespace

import torch

from attacker.data.dataset import episode_split


def test_episode_split():
    dataset = SimpleNamespace(
        split_episode_ids=torch.arange(40).repeat_interleave(5),
        sequence_indices=[(i, i + 1) for i in range(200)],
    )
    train_sets, validations = [], []
    for fraction in (0.1, 0.3, 1.0):
        train, val = episode_split(dataset, train_fraction=fraction)
        train_eps = {int(dataset.split_episode_ids[i]) for i in train.indices}
        val_eps = {int(dataset.split_episode_ids[i]) for i in val.indices}
        assert train_eps.isdisjoint(val_eps)
        assert len(train_eps) == max(1, int(38 * fraction))
        assert len(train.indices) == 5 * len(train_eps)
        train_sets.append(set(train.indices))
        validations.append(val.indices)
    assert train_sets[0] < train_sets[1] < train_sets[2]
    assert validations[0] == validations[1] == validations[2]
    assert episode_split(dataset)[0].indices == episode_split(dataset, train_fraction=1)[0].indices
    for fraction in (0, -0.1, 1.1):
        try:
            episode_split(dataset, train_fraction=fraction)
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid fraction: {fraction}")
    print("Nested episode subsets and fixed validation split verified.")


if __name__ == "__main__":
    test_episode_split()
