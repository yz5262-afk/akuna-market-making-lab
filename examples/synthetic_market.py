from pathlib import Path
import sys,json,random
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from market_maker import MarketMaker,Underlying,MarketHistory,BinaryOption,OptionLeg

random.seed(5262)
initial=[Underlying('FED',1,2.),Underlying('AJR',2,100.),Underlying('THR',3,105.)]
maker=MarketMaker(initial,[],1000.)
prior=maker.estimated_parameters
state={u.underlying_id:u.value for u in initial}
history={i:[v] for i,v in state.items()}
for _ in range(400):
    state=prior.advance_step(state)
    for i,v in state.items():history[i].append(v)
maker.warm_up(MarketHistory({i:tuple(v) for i,v in history.items()}))
assert maker.estimated_parameters != prior,'Synthetic calibration unexpectedly used fallback parameters'
strikes=np.linspace(85,115,31)
prices=[]
for strike in strikes:
    option=BinaryOption((OptionLeg(2,1.),),1,5,float(strike))
    prices.append(maker.price_option(option))
assert all(a>=b for a,b in zip(prices,prices[1:]))
option=BinaryOption((OptionLeg(2,1.),),1,5,100.)
rows=[]
for inventory in range(-20,21):
    maker.position.option_quantity_by_option_id[1]=inventory
    quote=maker.quote(option,0)
    rows.append({'inventory':inventory,'bid':quote.bid_price,'offer':quote.offer_price})
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
axes[0].plot(strikes,prices,color='#157a83',lw=2)
axes[0].set(xlabel='Binary option strike',ylabel='Simulated payoff probability',title='Five-step synthetic contract')
axes[1].plot([r['inventory'] for r in rows],[r['bid'] for r in rows],label='Bid',color='#157a83')
axes[1].plot([r['inventory'] for r in rows],[r['offer'] for r in rows],label='Offer',color='#d69b3a')
axes[1].set(xlabel='Current option inventory',ylabel='Quoted price',title='Inventory-aware two-sided quotes')
axes[1].legend()
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/strategy-diagnostics.svg')
out={'seed':5262,'synthetic_history_days':401,'calibration_changed_prior':True,'strikes':strikes.tolist(),'prices':prices,'quotes':rows,'performance_claim':'none; synthetic diagnostics only'}
(ROOT/'figures/results.json').write_text(json.dumps(out,indent=2))
print('Synthetic calibration, monotone strike curve and inventory quote diagnostics completed.')
