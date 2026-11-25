"""Read-only EXPLAIN ANALYZE benchmark of analytical views on current data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import psycopg

from autopyro.database.connection import local_parameters


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=int, default=45)
    args = parser.parse_args()
    if args.repeats<1 or args.timeout_seconds<1:
        parser.error('repeats and timeout must be positive')
    with psycopg.connect(**local_parameters(),connect_timeout=8) as conn:
        conn.read_only = True
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        conn.execute("SELECT set_config('statement_timeout',%s,true)",(str(args.timeout_seconds*1000),))
        sample = conn.execute("SELECT sample_id FROM geochem.analysis_run WHERE group_name LIKE 'Pyrol%' GROUP BY sample_id ORDER BY count(*) DESC,sample_id LIMIT 1").fetchone()[0]
        well = conn.execute("""SELECT s.well_id FROM geochem.sample s
            WHERE s.well_id IS NOT NULL AND s.top_md>=1000 AND s.top_md<3000
            AND EXISTS (SELECT FROM geochem.analysis_run a WHERE a.sample_id=s.sample_id AND a.group_name LIKE 'Pyrol%')
            GROUP BY s.well_id ORDER BY count(*) DESC,s.well_id LIMIT 1""").fetchone()[0]
        queries = {
            'measurement_sample': ('SELECT * FROM geochem.v_geochemical_measurement WHERE sample_id=%s',(sample,)),
            'measurement_well_depth': ('SELECT * FROM geochem.v_geochemical_measurement WHERE well_id=%s AND top_md>=1000 AND top_md<3000',(well,)),
            'pyrolysis_sample': ('SELECT * FROM geochem.v_pyrolysis_report WHERE sample_id=%s',(sample,)),
            'pyrolysis_well_depth': ('SELECT * FROM geochem.v_pyrolysis_report WHERE well_id=%s AND top_md>=1000 AND top_md<3000',(well,)),
            'pyrolysis_all': ('SELECT * FROM geochem.v_pyrolysis_report',()),
            'runs_sample': ('SELECT * FROM geochem.v_geochemical_run WHERE sample_id=%s',(sample,)),
            'runs_all': ('SELECT * FROM geochem.v_geochemical_run',()),
            'availability_well': ('SELECT * FROM geochem.v_geochemical_availability WHERE well_id=%s',(well,)),
            'availability_all': ('SELECT * FROM geochem.v_geochemical_availability',()),
            'quality_summary': ('SELECT * FROM geochem.v_geochemical_quality_summary',()),
            'lithostrat_status': ('SELECT * FROM geochem.v_sample_lithostrat_status',()),
        }
        report = {'created_at':datetime.now(timezone.utc).isoformat(),
                  'postgres':conn.execute('SELECT version()').fetchone()[0],
                  'sample_id':sample,'well_id':well,'repeats':args.repeats,
                  'settings':dict(conn.execute("SELECT name,setting FROM pg_settings WHERE name IN ('work_mem','jit','shared_buffers','max_parallel_workers_per_gather')")),
                  'counts':dict(conn.execute("SELECT 'results',count(*) FROM geochem.analysis_result UNION ALL SELECT 'runs',count(*) FROM geochem.analysis_run UNION ALL SELECT 'samples',count(*) FROM geochem.sample")),
                  'queries':{}}
        for name,(query,params) in queries.items():
            runs = []
            error = None
            for _ in range(args.repeats):
                try:
                    with conn.transaction():
                        plan = conn.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) '+query,params).fetchone()[0][0]
                        runs.append(plan)
                except psycopg.errors.QueryCanceled:
                    error = f'Timeout after {args.timeout_seconds}s'
                    break
            times = [p['Execution Time'] for p in runs]
            report['queries'][name] = {'sql':query,'params':params,'execution_ms':times,
                'median_ms':statistics.median(times) if times else None,'error':error,'plans':runs}
            print(f'{name}: {report["queries"][name]["median_ms"]} ms {error or ""}',flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
