"""Compare benchmark JSON reports; exit 1 on significant regressions, 2 on invalid comparisons."""
import argparse
import json
import math
from pathlib import Path
import statistics


def timings(query, repeats, name):
    values = query.get('execution_ms')
    plans = query.get('plans')
    if not isinstance(values,list) or len(values)!=repeats or not isinstance(plans,list) or len(plans)!=repeats:
        raise ValueError(f'{name}: incomplete repetitions')
    for value,plan in zip(values,plans):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError(f'{name}: invalid execution time')
        if value!=plan.get('Execution Time'):
            raise ValueError(f'{name}: timing does not match EXPLAIN')
    rows = [p.get('Plan',{}).get('Actual Rows') for p in plans]
    if any(type(n) not in (int,float) or not math.isfinite(n) or n<0 for n in rows) or len(set(rows))!=1:
        raise ValueError(f'{name}: inconsistent row counts')
    return statistics.median(values),rows[0]


def compare_reports(baseline, candidate, *, ratio=1.25, min_increase_ms=20):
    if not math.isfinite(ratio) or ratio<=1 or not math.isfinite(min_increase_ms) or min_increase_ms<0:
        raise ValueError('ratio must exceed 1; minimum increase must be finite and nonnegative')
    for key in ['postgres','settings','counts','sample_id','well_id']:
        if key not in baseline or key not in candidate or baseline[key]!=candidate[key]:
            raise ValueError(f'Incomparable reports: {key} differs or is missing')
    for report in [baseline,candidate]:
        if type(report.get('repeats')) is not int or report['repeats']<3:
            raise ValueError('At least three repetitions per report are required')
        if not isinstance(report.get('queries'),dict) or not report['queries']:
            raise ValueError('Missing benchmark queries')
    if baseline['queries'].keys()!=candidate['queries'].keys():
        raise ValueError('Incomparable reports: query sets differ')
    results = {}
    for name, before in baseline['queries'].items():
        after = candidate['queries'][name]
        for key in ['sql','params']:
            if key not in before or key not in after or before[key]!=after[key]:
                raise ValueError(f'{name}: query {key} differs or is missing')
        if before.get('error'):
            raise ValueError(f'{name}: baseline query failed')
        old,old_rows = timings(before,baseline['repeats'],name)
        if after.get('error'):
            results[name] = {'status':'failed','error':after['error'],'baseline_ms':old}
            continue
        new,new_rows = timings(after,candidate['repeats'],name)
        if old_rows!=new_rows:
            raise ValueError(f'{name}: returned row count changed ({old_rows} -> {new_rows})')
        delta = new-old
        regressed = new>old*ratio and delta>min_increase_ms
        results[name] = {'status':'regression' if regressed else 'ok',
                         'baseline_ms':old,'candidate_ms':new,'increase_ms':delta,
                         'ratio':new/old if old else None,'rows':old_rows}
    return {'ratio_threshold':ratio,'minimum_increase_ms':min_increase_ms,
            'passed':all(r['status']=='ok' for r in results.values()),'queries':results}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--ratio',type=float,default=1.25)
    parser.add_argument('--min-increase-ms',type=float,default=20)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() in (args.baseline.resolve(),args.candidate.resolve()):
        parser.error('Output must not overwrite either input report')
    try:
        result = compare_reports(json.loads(args.baseline.read_text(encoding='utf-8-sig')),
                                 json.loads(args.candidate.read_text(encoding='utf-8-sig')),
                                 ratio=args.ratio,min_increase_ms=args.min_increase_ms)
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        parser.exit(2,f'Comparison rejected: {exc}\n')
    for name,row in result['queries'].items():
        if row['status']=='failed':
            print(f'{name}: FAILED ({row["error"]})')
        else:
            print(f'{name}: {row["status"]} {row["baseline_ms"]:.3f} -> {row["candidate_ms"]:.3f} ms')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result['baseline_file'] = str(args.baseline)
        result['candidate_file'] = str(args.candidate)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
