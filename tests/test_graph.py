"""Run with: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
import torch
from asg2.graph import adjacency,RelationalNet

class GraphChecks(unittest.TestCase):
    def test_weighted_rows_and_empty_relation(self):
        a=adjacency([0,0,2],[0,1,1],[1.,3.,2.],4,3)
        np.testing.assert_allclose(a.to_dense().numpy().sum(1),[1,0,1,0])
        np.testing.assert_allclose(a.to_dense().numpy()[0,:2],[.25,.75])
        z=adjacency([],[],[],3,2)
        self.assertTrue(torch.isfinite(torch.sparse.mm(z,torch.ones(2,4))).all())
        self.assertEqual(float(torch.sparse.mm(z,torch.ones(2,4)).sum()),0)
    def example(self):
        a=adjacency([0,1],[0,1],[1,1],2,2)
        peer=adjacency([0,1],[1,0],[1,1],2,2)
        return {'xu':torch.ones(2,4),'xr':torch.ones(2,3),'ur':[a]*6,'ru':[a]*6,'peers':[peer]*6}
    def test_gradients_flow_through_graph(self):
        model=RelationalNet(4,3,hidden=8,use_graph=True)
        for p in model.parameters():torch.nn.init.constant_(p,.1)
        model.eval();model(self.example()).sum().backward()
        self.assertGreater(float(model.to_repo[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.to_user[0].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.peer[0].weight.grad.abs().sum()),0)
    def test_no_graph_ablation_ignores_edges(self):
        torch.manual_seed(42);model=RelationalNet(4,3,8,use_graph=False);model.eval()
        g=self.example();expected=model(g).detach()
        g['ur']=[];g['ru']=[];g['peers']=[]
        torch.testing.assert_close(expected,model(g))
    def test_messages_change_outputs(self):
        model=RelationalNet(4,3,8,use_graph=True)
        for p in model.parameters():torch.nn.init.constant_(p,.1)
        model.eval();g=self.example();a=model(g).detach()
        z=adjacency([],[],[],2,2);g['ur']=[z]*6;g['ru']=[z]*6;g['peers']=[z]*6
        self.assertFalse(torch.allclose(a,model(g)))

if __name__=='__main__':unittest.main()
