import pytest

from typhoon.evaluate_soft_moe_grid import simplex_grid


def test_simplex_grid_is_normalized_unique_and_includes_vertices():
    grid = simplex_grid(0.25)

    assert len(grid) == 15
    assert len(set(grid)) == len(grid)
    assert all(sum(point) == pytest.approx(1.0) for point in grid)
    assert (1.0, 0.0, 0.0) in grid
    assert (0.0, 1.0, 0.0) in grid
    assert (0.0, 0.0, 1.0) in grid


@pytest.mark.parametrize("step", [0.0, -0.25, 0.3, 2.0])
def test_simplex_grid_rejects_invalid_steps(step):
    with pytest.raises(ValueError):
        simplex_grid(step)
