"""Read-only audit of this specifically authorized normal training publication."""
from datetime import datetime, timezone
import hashlib,json,os,pickle,re,sqlite3,sys
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNFILES=(ROOT/'bazel-bin/python/stock_picker/api/main.runfiles').resolve()
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
sys.path[:0]=[str(ROOT/'python'),*map(str,RUNFILES.glob('rules_python++pip+*/site-packages'))]
import numpy as np
import pandas as pd
from stock_picker.features.dip_context import DIP_COLUMNS
from stock_picker.features.stacked_svm import STACKED_SVM_COLUMNS
from stock_picker.training.dataset import build_training_frame
from stock_picker.training.ensemble import predict_ensemble
from stock_picker.training.model import predict
from stock_picker.training.importance import model_type_importance
START='2026-11-09T17:31:06.433141-06:00'
PARENT='5deeea4d6ce248f1ade3592000575e12'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def get(path):
    with urlopen('http://127.0.0.1:8000'+path,timeout=60) as r:
        assert r.status==200
        return json.load(r)

def model(path):
    with path.open('rb') as f:
        return pickle.load(f)

def schema(trained):
    return {'model_type':trained.model_type,'feature_count':len(trained.feature_names),'feature_names':list(trained.feature_names),'dip_features':sorted(set(trained.feature_names)&set(DIP_COLUMNS))}

def redact_log_line(line):
    """Keep diagnostics while removing credential values before tracked output."""
    line=re.sub(r'(?i)(https?://)[^/\s:@]+:[^/\s@]+@',r'\1[REDACTED]@',line)
    line=re.sub(r'(?i)(authorization[\"\']?\s*[:=]\s*[\"\']?)(?:bearer|basic)\s+[^\s,;\'\"]+',r'\1[REDACTED]',line)
    keys=r'crumb|token|access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|x-api-key|authorization|auth|signature|sig|client[_-]?secret|password|credential|awsaccesskeyid'
    pattern=rf'(?i)(\b(?:{keys})[\"\']?\s*[:=]\s*)(?:\"[^\"]*\"|\'[^\']*\'|[^&\s\"\'<>;,]+)'
    return re.sub(pattern,r'\1[REDACTED]',line)

job=get('/api/training/status')
assert job['started_at']==START
assert job['status']=='completed' and job['error'] is None, job['status']
start=json.loads((HERE/'ordinary_training_start.json').read_text())
deployment=json.loads((HERE/'deployment_execution.json').read_text())
deployment_sources=deployment['source_sha256']
assert all(sha(ROOT/path)==expected for path,expected in deployment_sources.items()), 'Deployment source changed; review provenance before publication signoff.'
prune_audit=json.loads((HERE/'ordinary_prune_final.json').read_text())
prunes_path=ROOT/'data/pruned_features/pruned.json'
prunes=json.loads(prunes_path.read_text())
original=json.loads((HERE/'cache/deployment_before/pruned.json').read_text())
assert len(original)==93 and len(prunes)==95
old_by={r['feature']:r for r in original}
new_by={r['feature']:r for r in prunes}
assert all(new_by[k]==v for k,v in old_by.items())
removed={'dip_recovery_fraction','dip_volume_change_1d'}
assert set(new_by)-set(old_by)==removed
assert sha(prunes_path)==start['pruned_sha256']==prune_audit['final_sha256']
config_path=ROOT/'data/training_config/config.json'
config=json.loads(config_path.read_text())
assert config['included_features'] is None and config['model_choices'] is None and config['selected_run_id'] is None
assert sha(config_path)==start['configuration_sha256']
runs_path=ROOT/'data/training_runs/runs.json'
runs=json.loads(runs_path.read_text())
matching=[r for r in runs if r['started_at']==START]
assert len(matching)==1
record=matching[0]
assert record['status']=='completed' and record['error'] is None
assert max(runs,key=lambda r:r['started_at'])['run_id']==record['run_id']
assert record['date_range']==['2025-11-11','2026-11-09'],record['date_range']
assert 'MRCY' in record['train_tickers'] and 'MRCY' not in record['holdout_tickers']
assert [s['model_type'] for s in record['model_specs']]==['lightgbm','lightgbm_rank']
assert len(record['fold_metrics'])==4
fit_path=ROOT/'data/models/day_session_return.pkl'
rank_path=ROOT/'data/models/day_session_return_rank.pkl'
diag_path=ROOT/'data/models/day_session_return_logistic_diagnostic.pkl'
archive_path=ROOT/'data/models'/f"day_session_return_{record['run_id']}.pkl"
fit,rank,diag=map(model,(fit_path,rank_path,diag_path))
assert archive_path.exists() and sha(archive_path)==sha(fit_path)
active=set(DIP_COLUMNS)-removed
assert active==set(prune_audit['remaining_dip_inputs']) and len(active)==19
assert len(fit.members)==len(rank.members)==1
assert fit.members[0].model_type=='lightgbm' and rank.members[0].model_type=='lightgbm_rank'
assert len(fit.members[0].feature_names)==158
assert len(rank.members[0].feature_names)==151
svms=getattr(fit,'stacked_svm_estimators',{})
assert set(svms)==set(STACKED_SVM_COLUMNS) and len(svms)==7
raw=set(rank.members[0].feature_names)
assert len(raw)==151 and set(fit.members[0].feature_names)==raw|set(STACKED_SVM_COLUMNS)
all_models={**{f'fit_{i}':m for i,m in enumerate(fit.members)},**{f'rank_{i}':m for i,m in enumerate(rank.members)},'diagnostic':diag,**svms}
schemas={name:schema(m) for name,m in all_models.items()}
convergence={}
for name,m in {**svms,'diagnostic':diag}.items():
    steps=m.estimator.named_steps
    estimator=steps['regress'] if 'regress' in steps else steps['classify']
    iterations=np.atleast_1d(estimator.n_iter_).astype(int).tolist()
    convergence[name]={'estimator_class':type(estimator).__name__,'n_iter':iterations,'max_iter':int(estimator.max_iter),'reached_iteration_limit':bool(max(iterations)>=estimator.max_iter)}
for name,m in all_models.items():
    names=set(m.feature_names)
    assert names&set(DIP_COLUMNS)==active,(name,'dip schema mismatch')
    assert not names&set(new_by),(name,'pruned column present')
    if name not in {'fit_0'}:
        assert names==raw,(name,'raw schema mismatch')
assert set(record['resolved_features'])==raw|set(STACKED_SVM_COLUMNS)
with sqlite3.connect(f"file:{(ROOT/'data/mlruns/mlflow.db').resolve()}?mode=ro",uri=True) as con:
    con.row_factory=sqlite3.Row
    mlrun=dict(con.execute('SELECT * FROM runs WHERE run_uuid=?',(PARENT,)).fetchone())
    params={r['key']:r['value'] for r in con.execute('SELECT key,value FROM params WHERE run_uuid=?',(PARENT,))}
    folds=[dict(r) for r in con.execute("SELECT r.run_uuid,r.name,r.status,r.start_time,r.end_time FROM runs r JOIN tags t ON t.run_uuid=r.run_uuid AND t.key='mlflow.parentRunId' WHERE t.value=? ORDER BY r.start_time",(PARENT,))]
assert mlrun['status']=='FINISHED' and len(folds)==4 and all(r['status']=='FINISHED' for r in folds)
assert params['final_refit_through'].startswith('2026-11-09')
features=pd.read_parquet(ROOT/'data/features/MRCY.parquet')
prices=pd.read_parquet(ROOT/'data/prices/MRCY.parquet')
rows=build_training_frame(prices,features).tail(5)
assert set(DIP_COLUMNS)<=set(features.columns) and features.iloc[-1].dip_low_history_valid==1
assert len(features)==251 and str(features.index.max().date())=='2026-11-09'
smoke={}
for name,ensemble in [('fit',fit),('rank',rank)]:
    predictions=predict_ensemble(ensemble,rows)
    assert np.isfinite(predictions).all()
    smoke[name]={'rows':len(rows),'finite_predictions':True,'latest_prediction':float(predictions[-1])}
model_info=get('/api/model-info')
model_features=get('/api/model-features')
assert model_info['models'][0]['feature_count']==158 and model_info['rank_models'][0]['feature_count']==151
api_models={r['key']:r for r in model_features['models']}
assert set(api_models['fit']['features'])==raw|set(STACKED_SVM_COLUMNS)
assert set(api_models['rank']['features'])==raw
catalog=get('/api/catalog')
assert set(DIP_COLUMNS)<={name for group in catalog['catalog'].values() for name in group}
assert not set(DIP_COLUMNS)&set(catalog['experimental_features'])
with urlopen('http://127.0.0.1:5173/',timeout=30) as response:
    ui_status=response.status
assert ui_status==200
hashes={str(p.relative_to(ROOT)):sha(p) for p in (fit_path,rank_path,diag_path,archive_path)}
assert all(hashes[name]!=before for name,before in start['model_hashes_before'].items())
backups={}
for name,old_hash in start['model_hashes_before'].items():
    backup=HERE/'cache/deployment_before'/Path(name).name
    assert backup.exists() and sha(backup)==old_hash
    backups[str(backup.relative_to(ROOT))]=old_hash
lines=(HERE/'cache/deployment_logs/04_api.log').read_text(errors='replace').splitlines()
selected_log_lines=[(i+1,line) for i,line in enumerate(lines) if any(s in line for s in ('Warning','WARNING','Traceback','ERROR'))]
warning_lines=[{'line':number,'text':redact_log_line(line)} for number,line in selected_log_lines]
warning_summary={
    'convergence_warning_count':sum('ConvergenceWarning' in row['text'] for row in warning_lines),
    'liblinear_convergence_warning_count':sum('ConvergenceWarning' in row['text'] and 'Liblinear' in row['text'] for row in warning_lines),
    'other_convergence_warning_count':sum('ConvergenceWarning' in row['text'] and 'Liblinear' not in row['text'] for row in warning_lines),
    'other_python_warning_count':sum('Warning' in row['text'] and 'ConvergenceWarning' not in row['text'] for row in warning_lines),
    'concurrent_yfinance_error_line_count':sum('ERROR yfinance:' in row['text'] for row in warning_lines),
    'training_job_error':job['error'],
    'credential_redacted_line_count':sum(redact_log_line(line)!=line for _,line in selected_log_lines),
    'credential_redaction':'Credential query/config values and authorization headers are redacted before writing tracked excerpts; source API logs stay ignored.',
    'shared_log_note':'This is the shared API process log. Yahoo quote-request errors are not additional training convergence warnings; the job status and saved estimator iteration counts are recorded separately.',
}
tree_usage={}
for name,ensemble in [('fit',fit),('rank',rank)]:
    member=ensemble.members[0]
    native_names=list(member.estimator.feature_name())
    assert native_names==list(member.feature_names)
    raw_gains=dict(zip(native_names,member.estimator.feature_importance(importance_type='gain')))
    split_counts=dict(zip(native_names,member.estimator.feature_importance(importance_type='split')))
    percentages=model_type_importance(member)
    tree_usage[name]={column:{'included_as_input':True,'gain':float(raw_gains[column]),'gain_percent':float(percentages[column]),'split_count':int(split_counts[column]),'used_by_a_tree':bool(split_counts[column]>0)} for column in sorted(active)}
usage_rows=[{'feature':column,'fit':tree_usage['fit'][column],'rank':tree_usage['rank'][column],'svm_source_schema_count':sum(column in source.feature_names for source in svms.values())} for column in sorted(active)]
usage_report={
    'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_id':record['run_id'],
    'scope':'Descriptive read-only inspection of the published ordinary Fit/Rank artifacts with 19 new dip inputs added to 132 existing raw inputs (151 total raw inputs; 158 Fit-head inputs including seven SVM outputs); separate from schema correctness and predictive-performance evidence.',
    'model_hashes':{name:hashes[name] for name in (str(fit_path.relative_to(ROOT)),str(rank_path.relative_to(ROOT)))},
    'features':usage_rows,
    'summary':{name:{'included_dip_inputs':19,'dip_inputs_used_by_a_tree':sum(row['used_by_a_tree'] for row in values.values()),'total_dip_gain_percent':sum(row['gain_percent'] for row in values.values()),'zero_tree_split_features':[column for column,row in values.items() if not row['used_by_a_tree']]} for name,values in tree_usage.items()},
    'interpretation':['Gain is normalized within each model; raw gain magnitudes from Fit and Rank are not comparable because their objectives differ.','A saved input can have zero tree splits. All 19 new dip inputs remain in every one of the seven SVM source schemas, each containing 151 total raw inputs; schema inclusion does not by itself quantify their coefficient contribution.','Tree usage does not establish out-of-sample predictive benefit or justify a new prune. No fit, parameter change, or prune was performed.'],
    'existing_facility':'python/stock_picker/training/importance.py:model_type_importance',
    'avoided_mutating_endpoint':'GET /api/feature-importance calls highest_known_importance and records importance history; this audit did not call that endpoint.',
    'official_reference':json.loads((HERE/'operational_api_references.json').read_text())['lightgbm_importance'],
}
usage_json=HERE/'ordinary_dip_tree_usage.json'
usage_json.write_text(json.dumps(usage_report,indent=2,sort_keys=True)+'\n')
usage_text=['# Published ordinary dip inputs: tree usage','',f"Run `{record['run_id']}`. The 19 new dip inputs are added to 132 existing raw inputs: 151 total raw inputs for Rank and each of the seven SVM source models, and 158 Fit-head inputs including the seven SVM outputs. This table describes the new dip inputs only; it is not a performance comparison.",'','| Dip input | Fit gain % | Fit splits | Rank gain % | Rank splits |','|---|---:|---:|---:|---:|']
for row in usage_rows:
    usage_text.append(f"| `{row['feature']}` | {row['fit']['gain_percent']:.4f} | {row['fit']['split_count']} | {row['rank']['gain_percent']:.4f} | {row['rank']['split_count']} |")
usage_text.extend(['','Zero splits means the fitted trees did not split on that input. It does not remove the input from the seven SVM source models, demonstrate lack of predictive value, or trigger a prune. Gain percentages are normalized within each model; their raw gains use different objectives.','','Exact gains, model hashes, and API references: [ordinary_dip_tree_usage.json](ordinary_dip_tree_usage.json).'])
(HERE/'ordinary_dip_tree_usage.md').write_text('\n'.join(usage_text)+'\n')
report={
    'verified_at_utc':datetime.now(timezone.utc).isoformat(),'status':'verified','verification_source_sha256':sha(Path(__file__).resolve()),'runtime_versions':{'python':sys.version,'sqlite':sqlite3.sqlite_version,'numpy':np.__version__,'pandas':pd.__version__,'lightgbm':version('lightgbm'),'scikit_learn':version('scikit-learn')},'training_start_record_sha256':sha(HERE/'ordinary_training_start.json'),'api_start_record_sha256':sha(HERE/'api_restart_started.json'),
    'job':{k:v for k,v in job.items() if k!='result'},
    'run_record':{k:v for k,v in record.items() if k not in ('train_tickers','holdout_tickers','threshold_sweep')},
    'train_ticker_count':len(record['train_tickers']),'holdout_ticker_count':len(record['holdout_tickers']),
    'deployment_source_hashes_verified':deployment_sources,
    'mrcy_in_training':True,'configuration':config,'configuration_sha256':sha(config_path),
    'prune_verification':{'original_93_preserved':True,'total':95,'new_prunes':sorted(removed),'sha256':sha(prunes_path)},
    'schemas':schemas,'active_new_dip_inputs':sorted(active),'all_21_persisted_and_cataloged':True,
    'mlflow_parent':mlrun,'mlflow_params':params,'mlflow_folds':folds,
    'serving_model_hashes_before':start['model_hashes_before'],'serving_and_archive_hashes_after':hashes,
    'latest_fit_equals_run_archive':True,'recoverable_previous_models':backups,
    'model_info_response':model_info,'api_model_features_match_archives':True,
    'api_health':{'training_status':200,'model_info':200,'model_features':200,'catalog':200},'ui_status':ui_status,
    'mrcy_feature_rows':len(features),'mrcy_price_rows':len(prices),'mrcy_latest_low_history_valid':1,
    'read_only_prediction_smoke':smoke,'training_run_store_sha256':sha(runs_path),
    'warning_error_lines':warning_lines,'warning_summary':warning_summary,'final_estimator_iterations':convergence,'official_convergence_api_references':json.loads((HERE/'operational_api_references.json').read_text())['sklearn_convergence'],
    'limitations':['Completion and finite inference do not establish improved returns. The frozen research comparison with all 21 new dip inputs does not validate this production configuration with 19 new dip inputs, 151 total raw inputs, and 158 Fit-head inputs including seven SVM outputs.','Liblinear convergence warnings, if present, mean the optimizer did not certify convergence within its configured iteration budget; no hyperparameter changes or retraining were made during this audit.','The existing provider-quality limitations documented in input_anomaly_verification.json remain.','RunStore records Git HEAD at job completion; that field alone is not a hash of the code loaded when API PID24925 was started.'],
    'tree_usage_artifact':str(usage_json.relative_to(ROOT)),
    'mutations':'Only research verification/tree-usage artifacts; no API/model/prune/data changes and no Git staging.',
}
path=HERE/'ordinary_training_completion_verification.json'
path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'artifact':str(path.relative_to(ROOT)),'status':'verified','run_id':record['run_id'],'duration_seconds':record['duration_seconds'],'date_range':record['date_range'],'fit_head_inputs':158,'raw_inputs':151,'svm_source_models':len(svms),'active_new_dip_inputs':19,'warnings':len(warning_lines),'api_status':200,'ui_status':ui_status,'model_hashes_changed':True}))
