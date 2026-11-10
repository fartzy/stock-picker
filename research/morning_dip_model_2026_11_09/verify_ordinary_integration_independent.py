"""Independent ordinary-path parity against the frozen research panel."""
from pathlib import Path
import sys
import gzip
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUNFILES = ROOT / 'bazel-bin/python/stock_picker/api/main.runfiles'
sys.path[:0] = [str(ROOT/'python'), *map(str,RUNFILES.glob('rules_python++pip+*/site-packages'))]
import numpy as np
import pandas as pd
from stock_picker.features.dip_context import DIP_COLUMNS, DIP_OPEN_KNOWN_COLUMNS, build_dip_context_features, morning_dip_features
from stock_picker.features.pipeline import build_features
from stock_picker.features.main import feature_snapshot_window
from stock_picker.features.catalog import list_feature_columns, experimental_features
from stock_picker.features.registry import build_registry
from stock_picker.training.dataset import build_training_frame
from stock_picker.training.inference import build_inference_row
from stock_picker.training.model import feature_columns
from stock_picker.ingestion.history_policy import PRICE_HISTORY_PERIOD, FEATURE_SNAPSHOT_YEARS

HERE=ROOT/'research/morning_dip_model_2026_11_09'
SOURCE=ROOT/'research/averaging_down_2026_11_06/source_data/daily_five_year'
def load(ticker):
    bars=json.loads(gzip.decompress((SOURCE/f'{ticker}.json.gz').read_bytes()))['bars']
    return pd.DataFrame({name:[bar[key] for bar in bars] for name,key in zip(['Open','High','Low','Close','Volume'],['o','h','l','c','v'])},
        index=pd.to_datetime([datetime.fromtimestamp(bar['t']/1000,timezone.utc).date() for bar in bars]))

raw=load('MRCY')
spy=load('SPY')
assert raw.index.equals(spy.index), 'MRCY raw dates must match independent frozen SPY calendar'
frozen=pd.read_parquet(HERE/'dip_panel.parquet',filters=[('ticker','==','MRCY')]).set_index('date').sort_index().loc[:,list(DIP_COLUMNS)]
snapshots=build_dip_context_features(raw)
training=build_training_frame(raw,snapshots).loc[frozen.index,list(DIP_COLUMNS)]
pd.testing.assert_frame_equal(training,frozen,check_names=False,atol=1e-10,rtol=1e-10)
max_error=float((training-frozen).abs().max().max())
ordinary=build_features(raw)
ordinary_training=build_training_frame(raw,ordinary).loc[frozen.index,list(DIP_COLUMNS)]
pd.testing.assert_frame_equal(ordinary_training,frozen,check_names=False,atol=1e-10,rtol=1e-10)

# Missing-session expectations use the separately frozen SPY date index;
# they do not call the newly added calendar-completion helper.
gap_results=[]
for positions in ([600],[len(raw)-2],[600,len(raw)-2]):
    absent=raw.index[list(positions)]
    sparse=raw.drop(index=absent)
    canonical=sparse.reindex(spy.index)
    expected=morning_dip_features(canonical)
    feats=build_dip_context_features(sparse)
    aligned=build_training_frame(sparse,feats)
    pd.testing.assert_frame_equal(aligned[list(DIP_COLUMNS)],expected.loc[aligned.index],check_names=False,atol=1e-10,rtol=1e-10)
    previous=sparse.iloc[:-1]
    snapshot=build_dip_context_features(previous).iloc[-1]
    live=build_inference_row(snapshot,float(sparse.Open.iloc[-1]),float(previous.Close.iloc[-1]),
        previous.index[-1].date(),sparse.index[-1].date(),prior_history=previous).iloc[0][list(DIP_COLUMNS)]
    pd.testing.assert_series_equal(live,expected.iloc[-1],check_names=False,atol=1e-10,rtol=1e-10)
    old=pd.DataFrame({'existing_feature':np.arange(len(sparse),dtype=float)},index=sparse.index)
    old_only=build_training_frame(sparse,old)
    together=build_training_frame(sparse,pd.concat([old,feats],axis=1))
    pd.testing.assert_frame_equal(together[old_only.columns],old_only)
    gap_results.append({'removed':[str(x.date()) for x in absent],'compared_rows':len(aligned),
                        'live_low_history_valid':float(live.dip_low_history_valid)})

live_dates=[]
for t in [252,253,600,1000,1257,1258,1259]:
    previous=raw.iloc[:t]
    snapshot=build_dip_context_features(previous).iloc[-1]
    live=build_inference_row(snapshot,float(raw.Open.iloc[t]),float(previous.Close.iloc[-1]),previous.index[-1].date(),raw.index[t].date(),prior_history=previous).iloc[0][list(DIP_COLUMNS)]
    pd.testing.assert_series_equal(live,frozen.loc[raw.index[t]],check_names=False,atol=1e-10,rtol=1e-10)
    changed=raw.copy()
    changed.iloc[t:,changed.columns.get_indexer(['High','Low','Close','Volume'])]=[1e10,.01,1e8,1e12]
    changed.iloc[t+1:,changed.columns.get_loc('Open')]=1e7
    actual=build_training_frame(changed,build_features(changed)).loc[raw.index[t],list(DIP_COLUMNS)]
    pd.testing.assert_series_equal(actual,training.loc[raw.index[t]],check_names=False,atol=1e-10,rtol=1e-10)
    live_dates.append(str(raw.index[t].date()))

# Production history/retention exercise uses the same raw source only as an
# in-memory fixture; no production price file is written or substituted.
cutoff=raw.index[-1].date()
two_year=raw.loc[raw.index > pd.Timestamp(cutoff)-pd.DateOffset(years=2)]
full=build_features(two_year)
retained=feature_snapshot_window(full,cutoff)
assert PRICE_HISTORY_PERIOD=='2y' and FEATURE_SNAPSHOT_YEARS==1
assert retained.columns.equals(full.columns)
assert retained.index.min()>pd.Timestamp(cutoff)-pd.DateOffset(years=1)
assert retained.index.max()==pd.Timestamp(cutoff)
pd.testing.assert_frame_equal(retained,full.loc[retained.index])
assert len(retained)<len(two_year)
retained_training=build_training_frame(two_year,retained)
same_horizon=morning_dip_features(two_year).loc[retained_training.index]
pd.testing.assert_frame_equal(retained_training[list(DIP_COLUMNS)],same_horizon,check_names=False,atol=1e-10,rtol=1e-10)
full_coverage=retained_training.index[retained_training.dip_low_history_valid.eq(1)]
pd.testing.assert_frame_equal(retained_training.loc[full_coverage,list(DIP_COLUMNS)],frozen.loc[full_coverage],check_names=False,atol=1e-10,rtol=1e-10)
warmup_unknown_dates=retained_training.index[retained_training.dip_low_history_valid.eq(0)]

catalog=list_feature_columns(two_year)
assert catalog['dip_context']==list(DIP_COLUMNS)
assert not set(DIP_COLUMNS)&set(experimental_features())
views,services=build_registry(two_year)
assert any(v.name=='dip_context' and v.features==list(DIP_COLUMNS) for v in views)
assert all('dip_context' in service.feature_views for service in services)
selected=feature_columns(ordinary_training)
assert selected==list(DIP_COLUMNS)
excluded={'dip_deep_unrecovered','dip_open_to_low_252d'}
assert feature_columns(ordinary_training,excluded_features=excluded)==[c for c in DIP_COLUMNS if c not in excluded]

out={
 'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
 'frozen_panel_sha256':hashlib.sha256((HERE/'dip_panel.parquet').read_bytes()).hexdigest(),
 'mrcy_rows':len(frozen),'columns':len(DIP_COLUMNS),'ordinary_vs_frozen_cells':len(frozen)*len(DIP_COLUMNS),
 'max_absolute_error':max_error,'live_and_current_future_mutation_dates':live_dates,
 'missing_session_checks':gap_results,
 'all_existing_feature_values_labels_rows_preserved_in_gap_checks':True,
 'history_policy':{'price_period':PRICE_HISTORY_PERIOD,'snapshot_years':FEATURE_SNAPSHOT_YEARS,
   'input_rows':len(two_year),'retained_rows':len(retained),'training_rows':len(retained_training),
   'retained_from':str(retained.index.min().date()),'retained_through':str(retained.index.max().date()),
   'retained_columns':len(retained.columns),'all_families_computed_before_trimming':True,
   'earliest_rows_without_252_prior_sessions':[str(x.date()) for x in warmup_unknown_dates],
   'warmup_note':'Two calendar years do not guarantee 252 preceding sessions for every row in a retained calendar year; unavailable low history remains explicitly unknown, with validity 0.'},
 'ordinary_catalog_service_and_pruning':True,
 'source_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
 ROOT/'python/stock_picker/features/dip_context.py',ROOT/'python/stock_picker/features/pipeline.py',
 ROOT/'python/stock_picker/features/main.py',ROOT/'python/stock_picker/training/dataset.py',ROOT/'python/stock_picker/training/inference.py']},
 'production_writes':False,
}
Path('/tmp/dip-integration-numerical.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
