"""Independent scalar reconstruction from raw vendor bars, not feature helpers."""
from pathlib import Path
import sys
import gzip
import hashlib
import json
import math
import statistics as stats
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUNFILES = ROOT / 'bazel-bin/python/stock_picker/api/main.runfiles'
sys.path[:0] = [str(ROOT / 'python'), *map(str, RUNFILES.glob('rules_python++pip+*/site-packages'))]
import pandas as pd
import numpy as np
from stock_picker.features.dip_context import (
    morning_dip_features, morning_dip_row, DIP_COLUMNS, DIP_FAMILIES,
    DIP_FLAG_COLUMNS, DIP_OPEN_KNOWN_COLUMNS,
)

SOURCE = ROOT / 'research/averaging_down_2026_11_06/source_data/daily_five_year/MRCY.json.gz'
bars = json.loads(gzip.decompress(SOURCE.read_bytes()))['bars']
O, H, L, C, V = [[float(bar[name]) for bar in bars] for name in ('o','h','l','c','v')]
dates = [datetime.fromtimestamp(bar['t']/1000, timezone.utc).date() for bar in bars]
frame = pd.DataFrame({'Open':O, 'High':H, 'Low':L, 'Close':C, 'Volume':V}, index=pd.to_datetime(dates))
assert all(all(math.isfinite(x[i]) and x[i] > 0 for x in (O,H,L,C,V))
           and H[i]+1e-5 >= max(O[i],C[i]) and L[i]-1e-5 <= min(O[i],C[i]) and H[i]>=L[i]
           for i in range(len(bars)))

def manual(t):
    """List slices and standard-library statistics implement the frozen formulas."""
    p1, p2, p3 = C[t-1], C[t-2], C[t-3]
    low252 = min(L[t-252:t])
    older251 = min(L[t-252:t-1])
    age = t - max(i for i in range(t-252,t) if L[i] == low252)
    r1, r2 = p1/p2-1, p2/p3-1
    loss2 = p1/p3-1
    sigma = stats.stdev([C[i]/C[i-1]-1 for i in range(t-22,t-2)])
    recovery = (O[t]-p1)/(p3-p1) if p3 > p1 else float('nan')
    recovery_valid = math.isfinite(recovery)
    close_distance = p1/low252-1
    deep = (r1 <= -.02+1e-12 and r2 <= -.02+1e-12
            and -1e-12 <= close_distance <= .02+1e-12
            and -1e-12 <= recovery <= .25+1e-12) if recovery_valid else float('nan')
    tr1 = max(H[t-1]-L[t-1], abs(H[t-1]-p2), abs(L[t-1]-p2))
    tr2 = max(H[t-2]-L[t-2], abs(H[t-2]-p3), abs(L[t-2]-p3))
    location = (p1-L[t-1])/(H[t-1]-L[t-1]) if H[t-1]>L[t-1] else float('nan')
    pressure = (r1<0 and r2<0 and abs(r1)<abs(r2) and V[t-1]<V[t-2]
                and location>=.5-1e-12) if math.isfinite(location) else float('nan')
    vz = (V[t-1]-stats.mean(V[t-20:t]))/stats.stdev(V[t-20:t])
    support = min(L[t-21:t-1])
    sma, old_sma = stats.mean(C[t-50:t]), stats.mean(C[t-60:t-10])
    values = [
        close_distance, O[t]/low252-1, age, L[t-1]<=older251, O[t]<low252, 1.,
        loss2/(sigma*math.sqrt(2)), recovery, recovery_valid, deep,
        (abs(r1)-abs(r2))/sigma, tr1/tr2-1, V[t-1]/V[t-2]-1, pressure,
        (O[t]/p1-1)*vz, (support-L[t-1])/support, p1/support-1, O[t]/support-1,
        L[t-1]<support and p1>=support and O[t]>=support, sma/old_sma-1,
        loss2<=-.02+1e-12 and p1>sma and sma>old_sma,
    ]
    return {name:float(value) for name,value in zip(DIP_COLUMNS,values)}

computed = morning_dip_features(frame)
max_error = {name:0. for name in DIP_COLUMNS}
comparisons = 0
for t in range(252,len(frame)):
    expected = manual(t)
    actual = computed.iloc[t]
    for name,value in expected.items():
        got = float(actual[name])
        assert math.isnan(value) == math.isnan(got), (t,name,value,got)
        if math.isfinite(value):
            error = abs(value-got)
            max_error[name] = max(max_error[name],error)
            assert math.isclose(got,value,rel_tol=1e-10,abs_tol=1e-10), (t,name,value,got)
        comparisons += 1

selected = [252,253,600,1000,len(frame)-3,len(frame)-2,len(frame)-1]
for t in selected:
    scalar = morning_dip_row(frame.iloc[:t], O[t], dates[t])
    pd.testing.assert_series_equal(scalar,computed.iloc[t])
    revised = frame.copy()
    revised.iloc[t:, revised.columns.get_indexer(['High','Low','Close','Volume'])] = np.nan
    revised.iloc[t+1:, revised.columns.get_loc('Open')] = 999999
    pd.testing.assert_series_equal(morning_dip_features(revised).iloc[t],computed.iloc[t])
    revised = frame.copy()
    revised.iloc[t,revised.columns.get_loc('Open')] = O[t]*1.1
    stable = [name for name in DIP_COLUMNS if name not in DIP_OPEN_KNOWN_COLUMNS]
    pd.testing.assert_series_equal(morning_dip_features(revised).iloc[t][stable],computed.iloc[t][stable])

# A canonical missing session is preserved. It invalidates the 252-bar window
# until that exact row ages out; a shorter slice cannot stand in for 252 bars.
missing = frame.copy()
missing.iloc[100,:] = np.nan
missing_features = morning_dip_features(missing)
assert missing_features.iloc[252:353].dip_low_history_valid.eq(0).all()
assert missing_features.iloc[353].dip_low_history_valid == 1
assert missing_features.iloc[252:353].dip_prior_new_low_252d.isna().all()
assert missing_features.iloc[252:353].dip_deep_unrecovered.isna().all()
assert computed.iloc[251].dip_low_history_valid == 0
assert computed.iloc[252].dip_low_history_valid == 1

# Latest exact tie determines age, and the prior 252 sessions include yesterday.
synthetic = pd.DataFrame({'Open':101., 'High':102., 'Low':100., 'Close':101., 'Volume':1000.},
                         index=pd.date_range('2020-01-01',periods=254,freq='B'))
tie = morning_dip_features(synthetic).iloc[-1]
assert tie.dip_low_age_252d == 1 and tie.dip_prior_new_low_252d == 1
synthetic.iloc[-1,synthetic.columns.get_loc('Low')] = .01
assert morning_dip_features(synthetic).iloc[-1].dip_open_to_low_252d == tie.dip_open_to_low_252d

protocol = json.loads((ROOT/'research/morning_dip_model_2026_11_09/experiment_protocol.json').read_text())
assert protocol['families'] == {k:list(v) for k,v in DIP_FAMILIES.items()}
assert protocol['flags'] == list(DIP_FLAG_COLUMNS)
assert len(DIP_COLUMNS) == len(set(DIP_COLUMNS)) == 21
out = {
    'status':'passed', 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
    'source':str(SOURCE.relative_to(ROOT)), 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'module_sha256':hashlib.sha256((ROOT/'python/stock_picker/features/dip_context.py').read_bytes()).hexdigest(),
    'ticker':'MRCY', 'raw_bars':len(bars), 'tested_decision_rows':len(frame)-252,
    'decision_from':str(dates[252]), 'decision_through':str(dates[-1]),
    'cell_comparisons':comparisons, 'manual_method':'Raw JSON floats, scalar list slices, statistics.mean/stdev; no production rolling/formula helpers in expected values.',
    'max_absolute_errors':max_error, 'tolerance':{'absolute':1e-10,'relative':1e-10},
    'scalar_and_mutation_dates':[str(dates[t]) for t in selected],
    'current_future_hlcv_and_future_open_mutation':'all 21 current features unchanged on seven raw MRCY dates',
    'current_open_mutation':'all 13 prior-only columns unchanged on seven dates',
    'synthetic_checks':['252 bars required; 251 insufficient','internal missing row remains unknown for full window','low validity recovers exactly when missing bar exits','latest tied low has age one','today low excluded'],
    'protocol_schema':'21 unique names, four ordered families and four flags match exactly',
    'examples':{str(dates[t]):{k:(v if math.isfinite(v) else None) for k,v in manual(t).items()} for t in selected[-3:]},
}
Path('/tmp/dip-independent-numerical.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('examples','max_absolute_errors')}))
print('Maximum absolute difference:',max(max_error.values()))
