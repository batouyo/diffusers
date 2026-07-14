from pathlib import Path

import pytest

from scripts.run_parallel_evaluators import partition_paths


def test_partition_paths_is_disjoint_and_complete() -> None:
    paths = [Path(f"item-{index}") for index in range(23)]
    shards = partition_paths(paths, 5)
    flattened = [item for shard in shards for item in shard]
    assert len(flattened) == len(set(flattened)) == len(paths)
    assert set(flattened) == set(paths)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_partition_paths_rejects_invalid_count() -> None:
    with pytest.raises(ValueError):
        partition_paths([], 0)
