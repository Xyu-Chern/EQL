"""Small deterministic checks for the EQL draft; not RL training experiments."""
import json
import math
import random
import re
from pathlib import Path


def expectile(values, tau):
    assert values and 0 < tau < 1
    lo, hi = min(values), max(values)
    for _ in range(150):
        mid = (lo + hi) / 2
        residual = sum((tau if z >= mid else 1 - tau) * (z - mid) for z in values)
        if residual > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


tau, h = 0.25, 1e-6
y0 = expectile([0, 0], tau)
right = (expectile([h, -h], tau) - y0) / h
left = (expectile([-h, h], tau) - y0) / (-h)
assert abs(right + 0.5) < 1e-9 and abs(left - 0.5) < 1e-9

rng = random.Random(20261118)
two_errors, gradient_errors = [], []
for _ in range(1000):
    q = [rng.uniform(-10, 10) for _ in range(2)]
    t = rng.uniform(0.01, 0.99)
    mean = sum(q) / 2
    sd = abs(q[0] - q[1]) / 2  # population SD across the two members
    two_errors.append(abs(expectile(q, t) - (mean + (2 * t - 1) * sd)))
    q = [rng.uniform(-10, 10) for _ in range(5)]
    slopes = [rng.uniform(-2, 2) for _ in q]
    y = expectile(q, t)
    if min(abs(z - y) for z in q) < 1e-3:
        continue
    w = [t if z >= y else 1 - t for z in q]
    predicted = sum(wi * slope for wi, slope in zip(w, slopes)) / sum(w)
    fd = (expectile([z + h * slope for z, slope in zip(q, slopes)], t)
          - expectile([z - h * slope for z, slope in zip(q, slopes)], t)) / (2 * h)
    gradient_errors.append(abs(predicted - fd))
assert max(two_errors) < 1e-12
assert max(gradient_errors) < 1e-7

result = {
    'scope': 'Scalar math checks only; no manuscript implementation or RL performance tested.',
    'theorem_6_1_counterexample': {
        'Q1': 'a', 'Q2': '-a', 'tau': tau,
        'exact_expectile': '(2*tau-1)*abs(a)',
        'right_derivative': right, 'left_derivative': left,
        'draft_formula_at_zero': 0.0,
    },
    'two_critic_mean_sd_identity_max_error_1000_cases': max(two_errors),
    'non_tie_gradient_cases': len(gradient_errors),
    'non_tie_gradient_max_error': max(gradient_errors),
    'constant_ensemble': {'values': [10, 10], 'tau': tau, 'expectile': expectile([10, 10], tau),
                          'interpretation': 'If truth is 0, lower expectile still overestimates by 10.'},
    'trajectory_balance_counterexample': {'num_envs': 1, 'num_duplicates': 1, 'num_steps': 2,
        'num_critics': 2, 'transitions_per_bag_formula': 1,
        'issue': 'One indivisible two-step trajectory cannot fill two bags equally.'},
}
text_path = Path(__file__).resolve().parents[1] / 'tmp/pdfs/EQL.txt'
if text_path.exists():
    text = text_path.read_text(encoding='utf-8')
    terms = ['innovative', 'pioneering', 'revolutionary paradigm', 'transformative framework',
             'superior', 'surpass', 'excel', 'remarkable', 'unprecedented', 'breakthrough performance',
             'general-purpose', 'is capable of', 'notably', 'yet', 'yielding', 'at its essence',
             'encompass', 'differentiate', 'reveal', 'underscore', 'pave the way for',
             'highlight the potential of', 'profound challenges', 'stems from', 'rigid', 'impede']
    result['full_text_style_scan'] = {'em_dash_count': text.count('\u2014'),
        'term_matches': {term: len(re.findall(r'\b' + re.escape(term) + r'\b', text, re.I))
                         for term in terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.I)}}
output = Path(__file__).with_name('math_checks.json')
output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=True))
