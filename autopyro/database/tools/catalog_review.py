"""Export candidates for expert review; apply explicit decisions atomically."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from autopyro.database.connection import local_parameters


CATALOGS = {'indicator':('indicator','indicator_id','name'),
            'unit':('measurement_unit','unit_id','symbol'),
            'method':('analysis_method','method_id','name')}
STATE_SQL = '''SELECT jsonb_build_object('source_field',to_jsonb(f),'mapping',to_jsonb(s),
    'indicator',to_jsonb(i),'unit',to_jsonb(u),'method',to_jsonb(m))
    FROM geochem.source_field f
    LEFT JOIN geochem.source_field_semantics s USING(source_field_id)
    LEFT JOIN geochem.indicator i USING(indicator_id)
    LEFT JOIN geochem.measurement_unit u USING(unit_id)
    LEFT JOIN geochem.analysis_method m USING(method_id)
    WHERE f.target_table='analysis_result' ORDER BY f.source_field_id'''


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_document(path):
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=unique_keys)


def field_states(conn):
    return {row[0]['source_field']['source_field_id']:row[0] for row in conn.execute(STATE_SQL)}


def catalog_states(conn):
    result = {}
    for kind,(table,_,_) in CATALOGS.items():
        result[kind] = {row[0]['code']:row[0] for row in conn.execute(
            sql.SQL('SELECT to_jsonb(t) FROM geochem.{} t ORDER BY code').format(sql.Identifier(table)))}
    return result


def validate_document(doc):
    def text(value):
        return isinstance(value,str) and bool(value.strip())

    if not isinstance(doc,dict) or set(doc)!={'schema_version','reviewer','evidence','catalogs','mappings'}:
        raise ValueError('Document requires schema_version, reviewer, evidence, catalogs and mappings only')
    if type(doc['schema_version']) is not int or doc['schema_version']!=1:
        raise ValueError('Unsupported schema_version')
    if not text(doc['reviewer']) or not text(doc['evidence']):
        raise ValueError('Reviewer and evidence are required')
    if not isinstance(doc['catalogs'],list) or not isinstance(doc['mappings'],list) or not (doc['catalogs'] or doc['mappings']):
        raise ValueError('At least one explicit decision is required')
    seen = set()
    for entry in doc['catalogs']:
        if not isinstance(entry,dict) or not {'kind','code','expected'}<=set(entry) or set(entry)-{'kind','code','expected','label','description'}:
            raise ValueError('Invalid catalog decision keys')
        if not isinstance(entry['kind'],str) or entry['kind'] not in CATALOGS or not text(entry['code']):
            raise ValueError('Invalid catalog kind/code')
        key=(entry['kind'],entry['code'])
        if key in seen:
            raise ValueError('Duplicate catalog decision')
        seen.add(key)
        if entry['expected'] is None:
            if not text(entry.get('label')) or not text(entry.get('description')):
                raise ValueError('A new catalog entry requires label and description')
            if entry['code'].startswith(('source:','source-field:','source-group:')):
                raise ValueError('Source codes are reserved for imported provisional entries')
        elif not isinstance(entry['expected'],str) or not re.fullmatch('[0-9a-f]{64}',entry['expected']):
            raise ValueError('Expected catalog fingerprint is required')
    seen = set()
    for entry in doc['mappings']:
        if not isinstance(entry,dict) or set(entry)!={'source_field_id','expected','indicator_code','unit_code','method_code','note'}:
            raise ValueError('Invalid mapping decision keys')
        if type(entry['source_field_id']) is not int or entry['source_field_id']<1 or entry['source_field_id'] in seen:
            raise ValueError('Invalid or duplicate source_field_id')
        seen.add(entry['source_field_id'])
        if not isinstance(entry['expected'],str) or not re.fullmatch('[0-9a-f]{64}',entry['expected']):
            raise ValueError('Expected mapping fingerprint is required')
        if not text(entry['indicator_code']) or any(v is not None and not text(v) for v in [entry['unit_code'],entry['method_code']]) or not isinstance(entry['note'],str):
            raise ValueError('Invalid target codes or note')


def apply_review(conn,doc):
    """Caller owns commit/rollback. Lock review inputs before checking fingerprints."""
    validate_document(doc)
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('geochem:semantic-review',0))")
    conn.execute('''LOCK TABLE geochem.source_field,geochem.source_field_semantics,
        geochem.indicator,geochem.measurement_unit,geochem.analysis_method IN SHARE ROW EXCLUSIVE MODE''')
    digest=fingerprint(doc)
    previous=conn.execute('SELECT semantic_review_id FROM geochem.semantic_review_batch WHERE document_sha256=%s',(digest,)).fetchone()
    if previous:
        return {'review_id':previous[0],'already_applied':True}
    fields=field_states(conn)
    catalogs=catalog_states(conn)
    # Check every precondition before any catalog approval changes its fingerprint.
    for decision in doc['catalogs']:
        old=catalogs[decision['kind']].get(decision['code'])
        expected=fingerprint(old) if old is not None else None
        if decision['expected']!=expected:
            raise ValueError('Stale catalog decision: '+decision['code'])
        if old is not None:
            label=CATALOGS[decision['kind']][2]
            for key,column in [('label',label),('description','description')]:
                if key in decision and decision[key]!=old[column]:
                    raise ValueError('Do not rename a shared catalog entry; create a new canonical code')
    for decision in doc['mappings']:
        state=fields.get(decision['source_field_id'])
        if state is None or fingerprint(state)!=decision['expected']:
            raise ValueError('Stale or unknown source field: '+str(decision['source_field_id']))
    review_id=conn.execute('''INSERT INTO geochem.semantic_review_batch
        (document_sha256,reviewer,evidence,decision_document) VALUES (%s,%s,%s,%s)
        RETURNING semantic_review_id''',(digest,doc['reviewer'],doc['evidence'],Jsonb(doc))).fetchone()[0]

    def audit(kind,key,before,after):
        conn.execute('''INSERT INTO geochem.semantic_review_change
            (semantic_review_id,object_kind,object_key,before_state,after_state)
            VALUES (%s,%s,%s,%s,%s)''',(review_id,kind,str(key),Jsonb(before) if before is not None else None,Jsonb(after)))

    for decision in doc['catalogs']:
        kind,code=decision['kind'],decision['code']
        table,_,label=CATALOGS[kind]
        old=catalogs[kind].get(code)
        if old is None:
            query=sql.SQL('INSERT INTO geochem.{} (code,{},description,reviewed) VALUES (%s,%s,%s,true) RETURNING to_jsonb({})').format(
                sql.Identifier(table),sql.Identifier(label),sql.Identifier(table))
            new=conn.execute(query,(code,decision['label'],decision['description'])).fetchone()[0]
        else:
            query=sql.SQL('UPDATE geochem.{} SET reviewed=true WHERE code=%s RETURNING to_jsonb({})').format(sql.Identifier(table),sql.Identifier(table))
            new=conn.execute(query,(code,)).fetchone()[0]
        catalogs[kind][code]=new
        audit(kind,code,old,new)

    def resolve(kind,code):
        if code is None:
            return None
        target=catalogs[kind].get(code)
        if target is None:
            raise ValueError(f'Unknown {kind} code: {code}')
        return target[CATALOGS[kind][1]]

    for decision in doc['mappings']:
        field=decision['source_field_id']
        conn.execute('''INSERT INTO geochem.source_field_semantics
            (source_field_id,indicator_id,unit_id,method_id,reviewed,note) VALUES (%s,%s,%s,%s,true,%s)
            ON CONFLICT(source_field_id) DO UPDATE SET indicator_id=excluded.indicator_id,
                unit_id=excluded.unit_id,method_id=excluded.method_id,reviewed=true,note=excluded.note''',
            (field,resolve('indicator',decision['indicator_code']),resolve('unit',decision['unit_code']),
             resolve('method',decision['method_code']),decision['note']))
    updated=field_states(conn) if doc['mappings'] else {}
    for decision in doc['mappings']:
        field=decision['source_field_id']
        audit('mapping',field,fields[field],updated[field])
    return {'review_id':review_id,'already_applied':False,'catalog_decisions':len(doc['catalogs']),'mapping_decisions':len(doc['mappings'])}


def export_review(conn,directory):
    fields=field_states(conn)
    catalogs=catalog_states(conn)
    counts=dict(conn.execute('SELECT source_field_id,count(*) FROM geochem.analysis_result GROUP BY source_field_id'))
    groups=defaultdict(list)
    for field,state in fields.items():
        spec=state['source_field']
        groups[(spec['source_label'],spec['unit_label'],spec['source_dtype'])].append(field)
    candidates=[{'label':key[0],'unit_label':key[1],'dtype':key[2],'source_field_ids':ids,
                 'result_count':sum(counts.get(i,0) for i in ids),
                 'warning':'Names/units match exactly; scientific equivalence and methods are not confirmed'}
                for key,ids in groups.items() if len(ids)>1]
    candidates.sort(key=lambda c:(c['dtype']=='string',-c['result_count'],c['label']))
    directory.mkdir(parents=True,exist_ok=False)
    snapshot={'schema_version':1,'exported_at':datetime.now(timezone.utc).isoformat(),
              'fields':[{'source_field_id':field,'expected':fingerprint(state),'state':state,
                         'result_count':counts.get(field,0)} for field,state in fields.items()],
              'catalogs':{kind:[{'expected':fingerprint(state),'state':state} for state in values.values()]
                          for kind,values in catalogs.items()},'candidates':candidates}
    (directory/'catalog-snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'review-decisions.json').write_text(json.dumps({'schema_version':1,'reviewer':'','evidence':'','catalogs':[],'mappings':[]},ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Кандидаты для экспертной проверки','',
           f'Аналитических полей: {len(fields)}. Групп совпадающих названий/единиц/типов: {len(candidates)}.','',
           'Это кандидаты, а не подтверждённые эквивалентности. Методы и единицы не преобразуются.',
           'Текстовые поля могут быть метаданными, а не научными показателями.','',
           '| Исходное название | Тип | Полей | Результатов | Группы |','| --- | --- | ---: | ---: | --- |']
    for candidate in candidates:
        source_groups=', '.join(fields[i]['source_field']['group_name'] for i in candidate['source_field_ids'])
        escape=lambda value:str(value).replace('|','\\|').replace('\n',' ')
        lines.append(f'| {escape(candidate["label"])} | {candidate["dtype"]} | {len(candidate["source_field_ids"])} | {candidate["result_count"]} | {escape(source_groups)} |')
    (directory/'CANDIDATES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'fields':len(fields),'candidate_groups':len(candidates),'directory':str(directory.resolve())}


def main():
    parser=argparse.ArgumentParser(__doc__)
    actions=parser.add_subparsers(dest='action',required=True)
    export=actions.add_parser('export')
    export.add_argument('--output-dir',type=Path,required=True)
    apply=actions.add_parser('apply')
    apply.add_argument('document',type=Path)
    apply.add_argument('--commit',action='store_true',help='Commit reviewed decisions; otherwise rollback')
    args=parser.parse_args()
    if args.action=='apply':
        os.environ.setdefault('PGUSER','admin')
    try:
        with psycopg.connect(**local_parameters(),connect_timeout=8) as conn:
            conn.execute("SET LOCAL lock_timeout='15s'")
            if args.action=='export':
                conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
                result=export_review(conn,args.output_dir)
            else:
                result=apply_review(conn,read_document(args.document))
                if not args.commit:
                    conn.rollback()
                    result['dry_run']=True
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (OSError,ValueError,TypeError,KeyError,psycopg.Error) as exc:
        parser.exit(2,f'Review failed: {exc}\n')


if __name__=='__main__':
    main()
