"""Meaningful execution checks for the research engine."""
import unittest
import numpy as np
from simulation import Policy,simulate


class ExecutionTests(unittest.TestCase):
    def run_path(self, prices, policy, **kwargs):
        a=np.array(prices,dtype=float)
        return simulate(a,np.arange(len(a))*60.,np.ones(len(a),dtype=bool),policy,**kwargs)

    def test_flat_path_loses_round_trip_cost(self):
        r=self.run_path([[100,100,100,100,10000]]*3,Policy('flat'),cost=.001)
        self.assertAlmostEqual(r['pnl'],10000*.999/1.001-10000)

    def test_low_alone_does_not_trigger_close_confirmed_add(self):
        r=self.run_path([[100,101,90,100,10000],[100,101,99,100,10000]],Policy('ladder',max_tranches=4),cost=0)
        self.assertEqual(r['adds'],0)

    def test_add_at_next_open_not_prior_trigger(self):
        r=self.run_path([[100,100,94,94,10000],[93,96,92,96,10000]],Policy('ladder',max_tranches=4),cost=0)
        self.assertEqual(r['adds'],1)
        self.assertAlmostEqual(r['pnl'],(100+10000/93)*96-20000)

    def test_gap_cancels_add_and_stop_can_overshoot(self):
        r=self.run_path([[100,100,94,94,10000],[70,75,65,72,10000]],Policy('ladder',max_tranches=4,loss_limit=1000),cost=0)
        self.assertEqual(r['adds'],0)
        self.assertEqual(r['exit_reason'],'gap_stop')
        self.assertAlmostEqual(r['pnl'],-3000)

    def test_stop_has_priority_over_adding(self):
        r=self.run_path([[100,100,94,94,10000],[93,94,90,91,10000]],Policy('ladder',max_tranches=4,loss_limit=500),cost=0)
        self.assertEqual(r['adds'],0)
        self.assertEqual(r['exit_reason'],'stop')
        self.assertAlmostEqual(r['pnl'],-700)

    def test_rebound_above_add_limit_cancels_add(self):
        r=self.run_path([[100,100,94,94,10000],[97,99,96,98,10000]],Policy('ladder',max_tranches=4),cost=0)
        self.assertEqual(r['adds'],0)

    def test_future_bars_cannot_change_earlier_actions(self):
        a=[[100,100,94,94,10000],[94,95,90,90,10000],[90,91,86,87,10000],[86,90,85,89,10000]]
        b=a[:3]+[[110,115,108,112,10000]]
        p=Policy('ladder',max_tranches=4)
        ra=self.run_path(a,p,cost=0,keep_path=True)
        rb=self.run_path(b,p,cost=0,keep_path=True)
        self.assertEqual(ra['path'][:3],rb['path'][:3])

    def test_cap_and_half_exit(self):
        a=[[100,103,99,103,10000],[103,106,102,105,10000],[105,106,100,100,10000],[99,100,98,99,10000]]
        r=self.run_path(a,Policy('half',max_tranches=4,target=.02,partial_trail=True),cost=0)
        self.assertTrue(r['partial_exit'])
        self.assertEqual(r['exit_reason'],'trail')
        self.assertAlmostEqual(r['pnl'],100)
        self.assertLessEqual(r['max_capital'],40000)

    def test_overnight_gate_exits_at_next_open_before_gap(self):
        a=[[100,101,97,97,10000],[96,97,95,96,10000],[70,72,69,71,10000]]
        r=self.run_path(a,Policy('gate',overnight_loss_limit=200),cost=0,
                        overnight_check=np.array([True,False,False]))
        self.assertEqual(r['exit_reason'],'overnight')
        self.assertEqual(r['pnl'],-400)

    def test_overnight_gate_keeps_small_loss_and_blocks_add_on_large_loss(self):
        p=Policy('gate',max_tranches=4,overnight_loss_limit=600,overnight_loss_fraction=.03)
        small=self.run_path([[100,100,98,98,10000],[99,102,98,101,10000]],p,cost=0,
                            overnight_check=np.array([True,False]))
        large=self.run_path([[100,100,94,94,10000],[93,95,92,94,10000]],p,cost=0,
                            overnight_check=np.array([True,False]))
        self.assertEqual(small['exit_reason'],'time')
        self.assertEqual(large['exit_reason'],'overnight')
        self.assertEqual(large['adds'],0)


if __name__=='__main__':unittest.main()
