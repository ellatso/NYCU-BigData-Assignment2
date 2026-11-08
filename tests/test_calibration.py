import unittest
import numpy as np
import pandas as pd
import test_baselines
from asg2.calibration import eligible_origins,recency_weights,frequency12,blend

class CalibrationTests(unittest.TestCase):
    setUp=test_baselines.BaselineTests.setUp

    def test_labels_must_finish_before_four_hour_cutoff(self):
        origins=pd.to_datetime(['2026-08-05 12:00','2026-08-05 16:00','2026-08-12 16:00'])
        got=eligible_origins(origins,'2026-08-12 16:00','full_history')
        self.assertEqual(got,[pd.Timestamp('2026-08-05 12:00')])

    def test_recent_window_retains_only_last_twelve_origins(self):
        origins=list(pd.date_range('2026-01-01',periods=18,freq='7D'))
        self.assertEqual(eligible_origins(origins,'2026-08-01','recent12'),origins[-12:])

    def test_age_weights_have_six_week_half_life(self):
        w=recency_weights(pd.Series(pd.to_datetime(['2026-01-01','2026-02-12'])))
        self.assertAlmostEqual(w.mean(),1)
        self.assertAlmostEqual(w[0]/w[1],.5)

    def test_frequency_counts_weeks_and_blocks_future_availability(self):
        cutoff=pd.Timestamp('2026-09-23 20:00')
        base=self.events.iloc[[0]].copy();frames=[]
        for age in [1,2,8,84]:
            e=base.copy();e['event_time']=cutoff-pd.Timedelta(days=age);e['available_at']=e.event_time
            frames.append(e)
        late=base.copy();late['event_time']=cutoff-pd.Timedelta(days=15);late['available_at']=cutoff+pd.Timedelta(days=1)
        frames.append(late)
        f=frequency12(self.tables,pd.concat(frames),'2026-09-24',[7003,9001])
        self.assertEqual(f[0].sum(),0)
        self.assertAlmostEqual(f[1,-1],3/12)
        self.assertEqual(f[:,:-1].sum(),0)

    def test_blend_is_bounded_and_has_expected_endpoints(self):
        p=np.array([[.2,.8]]);f=np.array([[.6,.4]])
        np.testing.assert_allclose(blend(p,f,0),p)
        np.testing.assert_allclose(blend(p,f,1),f)
        np.testing.assert_allclose(blend(p,f,.25),[[.3,.7]])
        with self.assertRaises(ValueError):blend(p,f,1.1)

if __name__=='__main__':unittest.main()
