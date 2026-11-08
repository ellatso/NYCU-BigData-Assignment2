"""Post-review development diagnostics. Previously seen evaluation is not a test."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from asg2.core import TARGETS, score
from asg2.run import predict
from asg2.hybrid import last_week, weekly_score, apply_hybrid, select_switch, select_blend


def run(args):
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / 'protocol.json').read_text())
    manifest = json.loads((source / 'data_manifest.json').read_text())
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    cache = source / 'cache' / digest
    design = dict(calibration_origins=protocol['calibration_origins'],
                  evaluation_origins=protocol['holdout_origins'],
                  candidates=['temporal_switch', 'temporal_blend', 'dynamic_switch', 'dynamic_blend'],
                  selection='Mean weekly six-target Score on calibration only',
                  grid=dict(weights=[0, .25, .5, .75, 1], thresholds='Keep original thresholds as starting candidates; search 0.1 to 0.7 by 0.05'),
                  warning='Added after seeing Sep24/Oct1 results. All resulting evaluation comparisons are development diagnostics. Freeze any challenger for a future complete week.',
                  production_adoption='Do not automatically replace current model on reused evaluation results')
    (out / 'design.json').write_text(json.dumps(design, indent=2))

    def load(origins):
        samples = []
        for t in origins:
            key = pd.Timestamp(t).strftime('%Y%m%dT%H%M') + '_gap4.pkl'
            with (cache / key).open('rb') as f:
                samples.append(pickle.load(f))
        X = pd.concat([z[0] for z in samples], ignore_index=True)
        y = np.concatenate([z[1] for z in samples])
        folds = np.concatenate([np.repeat(i, len(z[0])) for i, z in enumerate(samples)])
        return X, y, folds

    C, cy, cf = load(protocol['calibration_origins'])
    rows, configs, calibrated = [], {}, {}
    for family, checkpoint in [('temporal', 'model_temporal.pkl'),
                               ('dynamic', 'model_dynamic_relational_mean_threshold.pkl')]:
        with (source / checkpoint).open('rb') as f:
            bundle = pickle.load(f)
        pc = predict(bundle['models'], C, bundle['columns'])
        rule = last_week(C)
        for method, selector in [('switch', select_switch), ('blend', select_blend)]:
            config, value = selector(cy, pc, rule, cf, bundle['thresholds'])
            name = family + '_' + method
            configs[name] = {**config, 'feature_family': family, 'checkpoint': checkpoint,
                             'calibration_mean_score': value}
            calibrated[name] = bundle
            rows.append(dict(candidate=name, split='calibration_mean', score=value))
            print('CALIBRATION', name, value, flush=True)
    # Persist the decision before the previously inspected evaluation is loaded.
    winner = max(configs, key=lambda n: configs[n]['calibration_mean_score'])
    (out / 'selection.json').write_text(json.dumps(dict(selected_challenger=winner,
        configs=configs, targets=TARGETS, production_replacement=False,
        reason='New recipes require a fresh forward comparison; reused results are diagnostic'), indent=2))
    D, dy, df = load(protocol['holdout_origins'])
    for name, config in configs.items():
        b = calibrated[name]
        prob = predict(b['models'], D, b['columns'])
        prediction = apply_hybrid(prob, last_week(D), config)
        rows.append(dict(candidate=name, split='pooled_diagnostic', **score(dy, prediction)))
        rows.append(dict(candidate=name, split='mean_diagnostic', score=weekly_score(dy, prediction, df)))
        for i, origin in enumerate(protocol['holdout_origins']):
            rows.append(dict(candidate=name, split=origin, **score(dy[df == i], prediction[df == i])))
        # Calibration-week stability: fit selection to one week, inspect the other.
        selector = select_switch if config['method'] == 'target_switch' else select_blend
        pc = predict(b['models'], C, b['columns'])
        for i in np.unique(cf):
            train = cf == i
            one, _ = selector(cy[train], pc[train], last_week(C)[train], cf[train], b['thresholds'])
            rows.append(dict(candidate=name, split=f'fit_calibration_week_{i}_evaluate_other',
                             **score(cy[~train], apply_hybrid(pc[~train], last_week(C)[~train], one))))
    for name, prediction in [('last_week_rule', last_week(D))]:
        rows.append(dict(candidate=name, split='pooled_diagnostic', **score(dy, prediction)))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    print('SELECTED CHALLENGER', winner, flush=True)
    print(pd.DataFrame(rows).query("split == 'pooled_diagnostic'").to_string(index=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='artifacts_coverage_20261102')
    p.add_argument('--out', default='artifacts_hybrid_20261102')
    run(p.parse_args())
