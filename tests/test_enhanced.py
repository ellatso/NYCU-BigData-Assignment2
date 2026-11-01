import unittest
import warnings
import numpy as np
import pandas as pd
import test_baselines
from asg2.core import build_events,features
from asg2.enhanced import enrich,push_training_mask,suspect_push_days

class EnhancedTests(unittest.TestCase):
    setUp=test_baselines.BaselineTests.setUp

    def test_thread_activity_excludes_self_and_future(self):
        self.tables['Issues']=pd.DataFrame(dict(issue_id=[0],repo_id=[0],author_id=[0],
            assignee_id=[None],pull_request_id=[None],is_pull_request=[False],
            created_at=pd.to_datetime(['2026-09-01']),first_seen_at=pd.to_datetime(['2026-09-01'])))
        self.tables['IssueComments']=pd.DataFrame(dict(comment_id=[0,1],issue_id=[0,0],repo_id=[0,0],
            author_id=[0,1],created_at=pd.to_datetime(['2026-09-20','2026-09-23'])))
        e=build_events(self.tables);base=features(self.tables,e,'2026-09-24',user_ids=[9001,7003])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            x=enrich(self.tables,e,base,'2026-09-24')
            self.assertAlmostEqual(x.v2r_thread_other3_IssueComments.iloc[0],np.log(2))
            self.assertEqual(x.v2r_thread_other3_IssueComments.iloc[1],0)
            self.assertEqual(x.v2_streak_weeks_push.iloc[0],1)
            self.assertEqual(x.v2_streak_weeks_push.iloc[1],0)
            f=self.tables['IssueComments'].iloc[[1]].copy();f['comment_id']=99
            f['created_at']=pd.Timestamp('2026-10-24')
            self.tables['IssueComments']=pd.concat([self.tables['IssueComments'],f],ignore_index=True)
            new=enrich(self.tables,build_events(self.tables),base,'2026-09-24')
        pd.testing.assert_frame_equal(x,new)

    def test_push_mask_is_target_week_local_and_end_exclusive(self):
        origins=pd.Series(pd.to_datetime(['2026-09-23 16:00','2026-09-30 16:00']))
        keep=push_training_mask(origins,[pd.Timestamp('2026-10-01')])
        self.assertEqual(keep.tolist(),[True,False])

    def test_suspect_uses_only_prior_reference_and_other_activity(self):
        rows=[]
        for day in pd.date_range('2026-01-01',periods=30):
            for _ in range(120):rows.append((day,'push'))
        rows.append((pd.Timestamp('2026-01-31'),'issue_comment'))
        e=pd.DataFrame(rows,columns=['event_time','kind'])
        self.assertIn(pd.Timestamp('2026-01-31'),suspect_push_days(e))

if __name__=='__main__':unittest.main()
