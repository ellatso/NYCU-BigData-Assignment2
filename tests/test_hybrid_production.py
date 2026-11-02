import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import test_baselines
from asg2.coverage import prediction_cutoffs
from asg2.hybrid import apply_hybrid, select_switch, select_blend, weekly_score
from asg2.run import final_predict, Constant


class HybridProductionTests(unittest.TestCase):
    def test_force_uses_observed_feature_clock_and_complete_training_cutoff(self):
        latest=pd.Timestamp('2026-11-10 12:00')
        with self.assertRaises(ValueError):prediction_cutoffs(latest)
        plan=prediction_cutoffs(latest,force=True)
        self.assertEqual(plan['training_cutoff'],latest)
        self.assertEqual(plan['feature_cutoff'],latest+pd.Timedelta(nanoseconds=1))
        self.assertTrue(plan['backup'])
        self.assertLess(plan['feature_cutoff'],plan['target'])

    def test_force_cannot_include_records_after_official_cutoff(self):
        plan=prediction_cutoffs('2026-11-20',force=True)
        self.assertEqual(plan['feature_cutoff'],pd.Timestamp('2026-11-11 12:00'))

    def test_calibration_hybrid_keeps_or_improves_starting_score(self):
        y=np.tile(np.array([[1,0,1,0,1,0],[0,1,0,1,0,1]]),(8,1))
        probability=np.full(y.shape,.2);rule=y.astype(bool);folds=np.repeat([0,1],8)
        initial=weekly_score(y,probability>=.5,folds)
        for selector in [select_switch,select_blend]:
            config,value=selector(y,probability,rule,folds,np.full(6,.5))
            self.assertGreater(value,initial)
            np.testing.assert_array_equal(apply_hybrid(probability,rule,config),y)

    def test_failed_format_validation_preserves_existing_final_file(self):
        fixture=test_baselines.BaselineTests();fixture.setUp()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);existing=out/'output_113700058.csv';existing.write_text('earlier valid submission')
            (out/'protocol.json').write_text(json.dumps(dict(selected_by_calibration_only='temporal',train_origins=['2026-05-06 16:00'])))
            with (out/'model_temporal.pkl').open('wb') as f:
                pickle.dump(dict(models=[Constant(.8)]*6,columns=['t_count7_push'],thresholds=np.full(6,.5)),f)
            roster=out/'input.csv';pd.DataFrame({'UserID':[7003,9001]}).to_csv(roster,index=False)
            args=SimpleNamespace(out=td,data=td,input=str(roster),force=True,no_refit=True,trees=1,
                                 push_policy_config=None,student_id='113700058')
            with patch('asg2.run.load_tables',return_value=fixture.tables),patch('asg2.run.fingerprint',return_value={}),patch('asg2.run.subprocess.run',side_effect=RuntimeError('invalid format')):
                with self.assertRaises(RuntimeError):final_predict(args)
            self.assertEqual(existing.read_text(),'earlier valid submission')
            self.assertEqual(list(out.glob('backup_*/output_*.csv')),[])
            self.assertEqual(list(out.glob('backup_*/.prediction_*.csv')),[])


if __name__=='__main__':unittest.main()
