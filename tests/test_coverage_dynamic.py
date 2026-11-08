import unittest
import numpy as np
import pandas as pd
import test_baselines
from asg2.core import build_events,features
from asg2.coverage import complete_training_origins,rank_predictions,push_frequency,push_status
from asg2.dynamic_relations import dynamic_features
from asg2.run import fit_models,predict

class CoverageDynamicTests(unittest.TestCase):
    setUp=test_baselines.BaselineTests.setUp

    def test_rank_ties_are_stable_and_counts_exact(self):
        p=np.ones((4,2));ids=[40,10,30,20]
        result=rank_predictions(p,[.5,.25],ids)
        self.assertEqual(result[:,0].tolist(),[False,True,False,True])
        self.assertEqual(result[:,1].tolist(),[False,True,False,False])

    def test_target_label_exclusion_only_changes_push_head(self):
        X=pd.DataFrame({'u_value':np.arange(30)})
        y=np.tile(np.r_[np.zeros(15),np.ones(15)][:,None],(1,6))
        normal=fit_models(X,y,['u_value'],trees=1)
        masked=fit_models(X,y,['u_value'],trees=1,target_masks={'push':np.arange(30)<15})
        p=predict(normal,X,['u_value']);q=predict(masked,X,['u_value'])
        np.testing.assert_array_equal(p[:,:5],q[:,:5]);np.testing.assert_array_equal(q[:,-1],0)

    def test_latest_complete_window_is_included_without_future_labels(self):
        cutoff=pd.Timestamp('2026-11-11 12:00')
        origins=complete_training_origins('2026-05-06 16:00',cutoff)
        self.assertEqual(origins[-1],pd.Timestamp('2026-11-03 16:00'))
        self.assertTrue(all(t+pd.Timedelta(days=7)<=cutoff for t in origins))

    def test_frozen_push_frequency_ignores_later_calendar_staleness(self):
        old=push_frequency(self.events,self.tables,'2026-09-24',[9001],True)
        later=push_frequency(self.events,self.tables,'2026-12-24',[9001],True)
        np.testing.assert_array_equal(old,later)
        self.assertTrue(push_status(self.events,'2026-09-24')['fallback_recommended'])
        last=self.events.loc[self.events.kind.eq('push'),'event_time'].max()
        self.assertTrue(push_status(self.events,last+pd.Timedelta(days=7),168)['fallback_recommended'])
        self.assertFalse(push_status(self.events,last+pd.Timedelta(days=7)-pd.Timedelta(seconds=1),168)['fallback_recommended'])

    def test_received_and_peer_activity_excludes_self_and_future(self):
        self.tables['Issues']=pd.DataFrame(dict(issue_id=[0],repo_id=[0],author_id=[0],assignee_id=[None],
            pull_request_id=[None],is_pull_request=[False],created_at=pd.to_datetime(['2026-09-01']),first_seen_at=pd.to_datetime(['2026-09-01'])))
        self.tables['IssueComments']=pd.DataFrame(dict(comment_id=[0,1],issue_id=[0,0],repo_id=[0,0],author_id=[0,1],created_at=pd.to_datetime(['2026-09-23','2026-09-23'])))
        self.tables['Pushes']=pd.DataFrame(dict(push_id=[0],repo_id=[0],user_id=[1],created_at=pd.to_datetime(['2026-09-23'])))
        events=build_events(self.tables);base=features(self.tables,events,'2026-09-24',user_ids=[9001,7003])
        x=dynamic_features(self.tables,events,base,'2026-09-24')
        self.assertAlmostEqual(x.d_received_7_IssueComments_received.iloc[0],np.log(2))
        self.assertAlmostEqual(x.d_peer7_push.iloc[0],np.log(2))
        future=self.tables['IssueComments'].iloc[[1]].copy();future['comment_id']=9;future['created_at']=pd.Timestamp('2026-10-01')
        self.tables['IssueComments']=pd.concat([self.tables['IssueComments'],future],ignore_index=True)
        z=dynamic_features(self.tables,build_events(self.tables),base,'2026-09-24')
        pd.testing.assert_frame_equal(x,z)

    def test_reviewed_repo_counts_only_visible_new_prs(self):
        self.tables['PullRequests']=pd.DataFrame(dict(pr_id=[0,1,2],repo_id=[0,0,0],author_id=[1,1,1],
            created_at=pd.to_datetime(['2026-09-01','2026-09-23','2026-09-23']),
            first_seen_at=pd.to_datetime(['2026-09-01','2026-09-23','2026-10-01'])))
        self.tables['PRReviews']=pd.DataFrame(dict(review_id=[0],pr_id=[0],repo_id=[0],reviewer_id=[0],
            submitted_at=pd.to_datetime(['2026-09-20']),first_seen_at=pd.to_datetime(['2026-09-20'])))
        events=build_events(self.tables);base=features(self.tables,events,'2026-09-24',user_ids=[9001,7003])
        x=dynamic_features(self.tables,events,base,'2026-09-24')
        self.assertAlmostEqual(x.d_reviewed_repo_new_pr3.iloc[0],np.log(2))
        self.assertEqual(x.d_reviewed_repo_new_pr3.iloc[1],0)

if __name__=='__main__':unittest.main()
