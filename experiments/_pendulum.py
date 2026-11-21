"""Read the common fixed-tau grid and independently selected SAC reference."""
from pathlib import Path

from eql.records import read

GRID_RESULTS = Path("results/pendulum_tau_grid")
SAC = Path("results/search_four/Pendulum-v1/SAC")

def grid_file(method, q, a, seed):
    return GRID_RESULTS / method / f"q{q}_a{a}_seed{seed}.json"


def reference(seed):
    run = read(SAC / f"confirm_2_{seed}.json")
    assert run["settings"] == {"learning_rate": 0.001} and run["seed"] == seed
    return run
