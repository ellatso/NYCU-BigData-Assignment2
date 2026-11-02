"""Small calibration-only mixtures of tree probabilities and last-week rules."""
import numpy as np
from .core import TARGETS, score


def last_week(frame):
    return np.column_stack([frame[f't_count7_{k}'].to_numpy() > 0 for k in TARGETS])


def weekly_score(y, prediction, folds):
    return float(np.mean([score(y[folds == f], prediction[folds == f])['score']
                          for f in np.unique(folds)]))


def apply_hybrid(probability, rule, config):
    if config['method'] == 'target_switch':
        return np.where(np.asarray(config['use_rule']), rule,
                        probability >= np.asarray(config['thresholds']))
    if config['method'] == 'probability_blend':
        weight = np.asarray(config['rule_weights'])
        return ((1 - weight) * probability + weight * rule >=
                np.asarray(config['thresholds']))
    raise ValueError('Unknown hybrid method')


def select_switch(y, probability, rule, folds, thresholds):
    """Optimize the coupled six-target Score; ties preserve the simpler policy."""
    config = dict(method='target_switch', thresholds=list(map(float, thresholds)),
                  use_rule=[False] * 6)
    best = weekly_score(y, apply_hybrid(probability, rule, config), folds)
    for _ in range(3):
        changed = False
        for j in range(6):
            candidate = {**config, 'use_rule': config['use_rule'].copy()}
            candidate['use_rule'][j] = not candidate['use_rule'][j]
            value = weekly_score(y, apply_hybrid(probability, rule, candidate), folds)
            if value > best + 1e-12:
                config, best, changed = candidate, value, True
        if not changed:
            break
    return config, best


def select_blend(y, probability, rule, folds, thresholds):
    """Fixed small grid, coordinate search, no evaluation labels accepted."""
    config = dict(method='probability_blend', rule_weights=[0.] * 6,
                  thresholds=list(map(float, thresholds)))
    best = weekly_score(y, apply_hybrid(probability, rule, config), folds)
    for _ in range(2):
        for j in range(6):
            for weight in [0., .25, .5, .75, 1.]:
                for threshold in np.linspace(.1, .7, 13):
                    candidate = {**config, 'rule_weights': config['rule_weights'].copy(),
                                 'thresholds': config['thresholds'].copy()}
                    candidate['rule_weights'][j] = float(weight)
                    candidate['thresholds'][j] = float(threshold)
                    value = weekly_score(y, apply_hybrid(probability, rule, candidate), folds)
                    if value > best + 1e-12:
                        config, best = candidate, value
    return config, best
