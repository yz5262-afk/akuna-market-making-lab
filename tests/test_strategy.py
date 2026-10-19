import unittest
from dataclasses import replace
import numpy as np
from market_maker import MarketMaker,Underlying,BinaryOption,OptionLeg,MarketHistory,FokOrder,OrderType

def maker():
    return MarketMaker([Underlying('FED',1,2.),Underlying('AJR',2,100.),Underlying('THR',3,105.)],[],1000.)

class StrategyChecks(unittest.TestCase):
    def test_expiry_payoff_boundary(self):
        m=maker()
        self.assertEqual(m.price_option(BinaryOption((OptionLeg(2,1.),),1,0,100.)),1.)
        self.assertEqual(m.price_option(BinaryOption((OptionLeg(2,1.),),1,0,101.)),0.)

    def test_shared_paths_imply_strike_monotonicity(self):
        m=maker()
        values=[m.price_option(BinaryOption((OptionLeg(2,1.),),1,4,k)) for k in [95.,100.,105.]]
        self.assertTrue(all(0<=p<=1 for p in values))
        self.assertGreater(values[0],values[-1])
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))

    def test_pricing_does_not_reset_global_rng(self):
        np.random.seed(123)
        before=np.random.get_state()
        maker().price_option(BinaryOption((OptionLeg(2,1.),),1,2,100.))
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])

    def test_zero_floor_uses_actual_rate_change(self):
        m=maker();m.underlying_state[0]=Underlying('FED',1,0.)
        p=replace(m.estimated_parameters,rate_up_probability=1e-12,rate_down_probability=1-1e-12,
            rate_reversion_strength=0.,ajarai_drift=0.,ajarai_idio_std_dev=0.,
            ajarai_sector_beta=0.,ajarai_rate_beta=1.,sector_std_dev=0.)
        self.assertEqual(m.price_option_from_parameters(p,BinaryOption((OptionLeg(2,1.),),1,1,90.)),1.)

    def test_inventory_skew_and_quote_invariants(self):
        m=maker();m.price_option=lambda option:.5
        o=BinaryOption((OptionLeg(2,1.),),1,2,100.)
        flat=m.quote(o,0)
        m.position.add_option_quantity(1,10)
        long=m.quote(o,0)
        self.assertLess(long.bid_price,flat.bid_price)
        self.assertLess(long.offer_price,flat.offer_price)
        for q in [flat,long]:
            self.assertTrue(0<=q.bid_price<q.offer_price<=1)
            self.assertAlmostEqual(q.bid_price*100,round(q.bid_price*100))
            self.assertAlmostEqual(q.offer_price*100,round(q.offer_price*100))

    def test_fok_inventory_cash_and_price_gates(self):
        m=maker();m.price_option=lambda option:.5
        o=BinaryOption((OptionLeg(2,1.),),1,2,100.)
        m.position.add_option_quantity(1,20)
        self.assertFalse(m.respond_to_fok(o,FokOrder(0,1,OrderType.SELL,.1,5)))
        self.assertTrue(m.respond_to_fok(o,FokOrder(0,1,OrderType.BUY,.9,5)))
        self.assertFalse(m.respond_to_fok(o,FokOrder(0,1,OrderType.BUY,.4,5)))
        m.cash_balance=m.MIN_CASH_BUFFER
        self.assertFalse(m.respond_to_fok(o,FokOrder(0,1,OrderType.BUY,.9,5)))

if __name__=='__main__':unittest.main()
