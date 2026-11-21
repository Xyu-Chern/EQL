"""Ensemble Q-learning algorithms and training components."""


def __getattr__(name):
    if name == "EQL":
        from eql.algorithms.eql import EQL
        return EQL
    if name == "compute_expectile_on_axis":
        from eql.expectiles import compute_expectile_on_axis
        return compute_expectile_on_axis
    raise AttributeError(name)
