"""Regression checks for official-ID mapping and availability-gated labels."""
import unittest
import numpy as np
import pandas as pd
from asg2.core import COLS, TABLES, build_events, features, labels

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tables={n:pd.DataFrame(columns=COLS[n]) for n in TABLES}
        for d in self.tables.values():
            for c in d:
                if c.endswith('_at'):d[c]=pd.to_datetime(d[c])
        self.tables['Issues']['is_pull_request']=pd.Series(dtype=bool)
        self.tables['Users']=pd.DataFrame(dict(user_id=[0,1],github_user_id=[9001,7003],
            is_cohort=[True,True],first_seen_at=pd.to_datetime(['2026-01-01']*2)))
        self.tables['Repositories']=pd.DataFrame(dict(repo_id=[0],first_seen_at=pd.to_datetime(['2026-01-01'])))
        self.tables['Pushes']=pd.DataFrame(dict(push_id=[0],repo_id=[0],user_id=[0],created_at=pd.to_datetime(['2026-09-20'])))
        self.events=build_events(self.tables)

    def test_explicit_ids_preserve_order_and_stable_identity(self):
        for ids in [[7003,9001],np.array([7003,9001]),pd.Series([7003,9001],name='UserID')]:
            x=features(self.tables,self.events,'2026-09-24',user_ids=ids)
            self.assertEqual(x.UserID.tolist(),[7003,9001])
            self.assertEqual(x.t_count7_push.iloc[0],0)
            self.assertGreater(x.t_count7_push.iloc[1],0)

    def test_future_availability_is_not_a_historical_label(self):
        e=self.events.copy();e['event_time']=pd.Timestamp('2026-09-25')
        e['available_at']=pd.Timestamp('2026-10-05')
        self.assertEqual(labels(self.tables,e,'2026-09-24',[9001]).sum(),0)
        e['available_at']=pd.Timestamp('2026-09-25')
        self.assertEqual(labels(self.tables,e,'2026-09-24',[9001])[0,-1],1)

if __name__=='__main__':unittest.main()
