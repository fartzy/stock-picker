"""Prepared post-training backup only; no live-file replacement or Git staging."""
from datetime import datetime, timezone
from contextlib import closing
import argparse,hashlib,json,sqlite3,sys
from pathlib import Path
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES=('data/jev/forecasts.db','data/mlruns/mlflow.db','data/paper_book/paper_book.db')
EXPECTED_START='2026-11-09T17:31:06.433141-06:00'

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def quote(name): return '"'+name.replace('"','""')+'"'
def check_job():
    with urlopen('http://127.0.0.1:8000/api/training/status',timeout=30) as response: job=json.load(response)
    assert job['started_at']==EXPECTED_START and job['status']=='completed' and job['error'] is None
    return {k:v for k,v in job.items() if k!='result'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--approved-after-audit',action='store_true')
    args=parser.parse_args()
    if not args.approved_after_audit: parser.error('Explicit root go-ahead is required after publication audit; preparation does not execute a backup.')
    assert sys.version_info[:2]==(3,12), 'Use the application hermetic Python runtime.'
    audit_path=HERE/'ordinary_training_completion_verification.json'
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='verified' and audit['job']['started_at']==EXPECTED_START
    job=check_job()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=HERE/'cache/final_sqlite_snapshots'/stamp
    output.mkdir(parents=True,exist_ok=False)
    manifest_path=HERE/'final_sqlite_snapshot_manifest.json'
    assert not manifest_path.exists(), 'Refuse to overwrite an earlier final snapshot manifest.'
    records=[]
    for relative in SOURCES:
        check_job()
        source=ROOT/relative
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        started=now()
        with closing(sqlite3.connect(f'file:{source.resolve()}?mode=ro',uri=True,timeout=30)) as origin:
            origin.execute('PRAGMA query_only=ON')
            journal_mode=origin.execute('PRAGMA journal_mode').fetchone()[0]
            origin.execute('BEGIN')
            origin.execute('SELECT count(*) FROM sqlite_schema').fetchone()
            try:
                with closing(sqlite3.connect(destination,timeout=30)) as target:
                    origin.backup(target,pages=256,sleep=0.1)
                    assert target.execute('PRAGMA journal_mode=DELETE').fetchone()[0]=='delete'
            finally:
                origin.rollback()
        completed=now()
        with closing(sqlite3.connect(f'file:{destination.resolve()}?mode=ro',uri=True)) as copy:
            integrity=[r[0] for r in copy.execute('PRAGMA integrity_check')]
            foreign_keys=copy.execute('PRAGMA foreign_key_check').fetchall()
            assert integrity==['ok'], (relative,integrity)
            assert not foreign_keys, (relative,'foreign key violations',foreign_keys[:5])
            schema=copy.execute("SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name").fetchall()
            tables=[r[0] for r in copy.execute("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name")]
            counts={table:copy.execute(f'SELECT count(*) FROM {quote(table)}').fetchone()[0] for table in tables}
        for suffix in ('-wal','-journal'):
            sidecar=Path(str(destination)+suffix)
            assert not sidecar.exists() or sidecar.stat().st_size==0,(relative,'nonempty snapshot sidecar')
        destination.chmod(0o400)
        records.append({'live_path':relative,'snapshot_path':str(destination.relative_to(ROOT)),'started_at_utc':started,'completed_at_utc':completed,'source_journal_mode':journal_mode,'snapshot_journal_mode':'delete','integrity_check':integrity,'foreign_key_violations':0,'bytes':destination.stat().st_size,'sha256':sha(destination),'schema_sha256':hashlib.sha256(json.dumps(schema).encode()).hexdigest(),'table_row_counts':counts})
    report={'status':'verified','created_at_utc':now(),'snapshot_source_sha256':sha(Path(__file__).resolve()),'python_version':sys.version,'sqlite_version':sqlite3.sqlite_version,'job':job,'publication_audit_path':str(audit_path.relative_to(ROOT)),'publication_audit_sha256':sha(audit_path),'api_references_path':str((HERE/'operational_api_references.json').relative_to(ROOT)),'api_references_sha256':sha(HERE/'operational_api_references.json'),'databases':records,'source_mutations':'No logical writes, replacement, journal-mode changes, checkpoints, vacuum, or Git staging. SQLite may create its own necessary lock/sidecar bookkeeping.','consistency':'Each database is a separate transaction-consistent snapshot including its committed WAL contents; the three files are not a cross-database atomic snapshot.','git_staging':'Pending root authorization/index coordination; stage these exact snapshot bytes via hash-object and update-index --cacheinfo, never git add of live DB files.'}
    manifest_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'manifest':str(manifest_path.relative_to(ROOT)),'status':'verified','databases':[{k:r[k] for k in ('live_path','snapshot_path','bytes','sha256')} for r in records]}))

if __name__=='__main__': main()
