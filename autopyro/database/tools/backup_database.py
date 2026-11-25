"""Create a full Compose database dump and verify restoration in an isolated DB."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

import psycopg
from psycopg import sql

from autopyro.database.connection import local_parameters


ROOT = Path(__file__).resolve().parents[3]


def docker(*args, stdin=None, stdout=None):
    result = subprocess.run(['docker','compose','exec','-T','postgres',*args],
                            cwd=ROOT,stdin=stdin,stdout=stdout or subprocess.PIPE,
                            stderr=subprocess.PIPE,check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8',errors='replace').strip())
    return result.stdout.decode('utf-8').strip() if stdout is None else None


def check_restore_name(name, source):
    if name==source or not re.fullmatch(r'geochem_restorecheck_[0-9a-f]{16}',name):
        raise ValueError('Refusing to operate on a database outside the generated restore-test namespace')


def manifest(conn):
    conn.execute("SET LOCAL timezone='UTC'")
    schemas = "n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%' AND n.nspname NOT LIKE 'pg_temp%'"
    tables = conn.execute(f"SELECT n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind IN ('r','p') AND {schemas} ORDER BY 1,2").fetchall()
    fingerprints = {}
    for schema,table in tables:
        # SHA256 of sorted per-row SHA256 digests includes duplicates and ignores row order.
        count,digest = conn.execute(sql.SQL('''SELECT count(*),encode(sha256(convert_to(
            coalesce(string_agg(digest,'' ORDER BY digest),''),'UTF8')),'hex')
            FROM (SELECT encode(sha256(convert_to(row_to_json(t)::text,'UTF8')),'hex') AS digest
                  FROM {}.{} t) rows''').format(sql.Identifier(schema),sql.Identifier(table))).fetchone()
        fingerprints[f'{schema}.{table}'] = {'rows':count,'sha256':digest}
    sequences = {}
    for schema,name in conn.execute(f"SELECT n.nspname,c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='S' AND {schemas} ORDER BY 1,2"):
        sequences[f'{schema}.{name}'] = list(conn.execute(sql.SQL('SELECT last_value,is_called FROM {}.{}').format(sql.Identifier(schema),sql.Identifier(name))).fetchone())
    definitions = {}
    statements = {
        'views':f"SELECT n.nspname,c.relname,pg_get_viewdef(c.oid,true) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='v' AND {schemas} ORDER BY 1,2",
        'constraints':f"SELECT n.nspname,c.relname,k.conname,pg_get_constraintdef(k.oid,true) FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE {schemas} ORDER BY 1,2,3",
        'indexes':f"SELECT n.nspname,c.relname,pg_get_indexdef(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind='i' AND {schemas} ORDER BY 1,2",
        'triggers':f"SELECT n.nspname,c.relname,t.tgname,pg_get_triggerdef(t.oid,true) FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE NOT t.tgisinternal AND {schemas} ORDER BY 1,2,3",
        'functions':f"SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid),pg_get_functiondef(p.oid) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE p.prokind='f' AND {schemas} ORDER BY 1,2,3",
        'columns':f"SELECT n.nspname,c.relname,a.attnum,a.attname,format_type(a.atttypid,a.atttypmod),a.attnotnull,a.attidentity,a.attgenerated,pg_get_expr(d.adbin,d.adrelid) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_namespace n ON n.oid=c.relnamespace LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum WHERE c.relkind IN ('r','p') AND a.attnum>0 AND NOT a.attisdropped AND {schemas} ORDER BY 1,2,3",
        'owners_acl':f"SELECT n.nspname,c.relname,c.relowner::regrole::text,c.relacl::text FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.relkind IN ('r','p','v','S') AND {schemas} ORDER BY 1,2",
        'schema_acl':f"SELECT n.nspname,n.nspowner::regrole::text,n.nspacl::text FROM pg_namespace n WHERE {schemas} ORDER BY 1",
        'function_acl':f"SELECT n.nspname,p.proname,pg_get_function_identity_arguments(p.oid),p.proowner::regrole::text,p.proacl::text FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE p.prokind='f' AND {schemas} ORDER BY 1,2,3",
        'default_acl':f"SELECT n.nspname,a.defaclrole::regrole::text,a.defaclobjtype,a.defaclacl::text FROM pg_default_acl a JOIN pg_namespace n ON n.oid=a.defaclnamespace WHERE {schemas} ORDER BY 1,2,3",
        'sequence_definitions':f"SELECT n.nspname,c.relname,format_type(s.seqtypid,NULL),s.seqstart,s.seqincrement,s.seqmax,s.seqmin,s.seqcache,s.seqcycle FROM pg_sequence s JOIN pg_class c ON c.oid=s.seqrelid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE {schemas} ORDER BY 1,2",
    }
    for name,statement in statements.items():
        definitions[name] = [list(row) for row in conn.execute(statement)]
    return {'tables':fingerprints,'sequences':sequences,'definitions':definitions}


def backup(output_root):
    os.environ.setdefault('PGUSER','admin')
    parameters = local_parameters()
    if parameters['dbname']!='geochemistry':
        raise ValueError('This command backs up only the Compose geochemistry database')
    token = uuid.uuid4().hex
    directory = output_root / ('geochemistry_'+datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')+'_'+token[:8])
    directory.mkdir(parents=True,exist_ok=False)
    archive = directory/'geochemistry.dump'
    temporary = 'geochem_restorecheck_'+token[:16]
    check_restore_name(temporary,parameters['dbname'])
    created = False
    report = {'status':'in_progress','created_at':datetime.now(timezone.utc).isoformat()}
    report_path = directory/'verification.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Backup directory: {directory.resolve()}',flush=True)
    try:
        with psycopg.connect(**parameters,connect_timeout=8) as source:
            source.read_only = True
            source.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            cluster = str(source.execute('SELECT system_identifier FROM pg_control_system()').fetchone()[0])
            local_cluster = docker('psql','-X','-At','-U','admin','-d','geochemistry','-c','SELECT system_identifier FROM pg_control_system()')
            if cluster!=local_cluster:
                raise ValueError('Connection and Compose service belong to different PostgreSQL clusters')
            snapshot = source.execute('SELECT pg_export_snapshot()').fetchone()[0]
            print('Creating custom-format dump from a shared snapshot...',flush=True)
            with archive.open('wb') as stream:
                docker('pg_dump','-U','admin','--dbname=geochemistry','--format=custom','--snapshot='+snapshot,stdout=stream)
            with archive.open('rb') as stream:
                report['sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
            report['bytes'] = archive.stat().st_size
            (directory/'SHA256SUMS.txt').write_text(report['sha256']+'  geochemistry.dump\n',encoding='utf-8')
            print('Fingerprinting source snapshot...',flush=True)
            original = manifest(source)
            print('Restoring isolated database...',flush=True)
            docker('createdb','-U','admin','--template=template0','--owner=admin',temporary)
            created = True
            with archive.open('rb') as stream:
                docker('pg_restore','-U','admin','--exit-on-error','--single-transaction','--dbname='+temporary,stdin=stream)
            restored_parameters = {**parameters,'dbname':temporary}
            with psycopg.connect(**restored_parameters,connect_timeout=8) as restored:
                restored.read_only = True
                restored.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
                recovered = manifest(restored)
            report['source'] = original
            report['restored'] = recovered
            if original!=recovered:
                raise ValueError('Restored data/schema manifest differs from source snapshot')
            # Sequence state is not MVCC-snapshotted: reject a moving source.
            for name,state in original['sequences'].items():
                schema,sequence = name.split('.',1)
                current = list(source.execute(sql.SQL('SELECT last_value,is_called FROM {}.{}').format(sql.Identifier(schema),sql.Identifier(sequence))).fetchone())
                if current!=state:
                    raise ValueError('Source sequences changed during verification; retry without concurrent writes')
            report['status']='verified'
    except BaseException as exc:
        report['status']='failed'
        report['error']=str(exc)
        raise
    finally:
        if created:
            check_restore_name(temporary,parameters['dbname'])
            try:
                docker('dropdb','-U','admin',temporary)
                report['temporary_database_removed']=True
            except BaseException:
                report['status']='cleanup_failed'
                report['temporary_database']=temporary
                raise
            finally:
                report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'README.md').write_text(
        '# Проверенный дамп geochemistry\n\n'
        f'Статус: verified. Размер: {report["bytes"]} байт.\n\n'
        f'Таблиц: {len(original["tables"])}; последовательностей: {len(original["sequences"])}.\n\n'
        'Восстановление проверено в отдельной базе; временная база удалена. '
        'Исходные данные, определения объектов, владельцы и ACL совпали.\n\n'
        'Серверные роли, их пароли, конфигурация PostgreSQL и локальный .env '
        'в архив не входят. На другом сервере предварительно создайте admin и readonly.\n\n'
        'Подробности: ../../BACKUPS.md; результаты: verification.json.\n',encoding='utf-8')
    print(f'Verified backup: {directory.resolve()}',flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'database/backups')
    args = parser.parse_args()
    try:
        backup(args.output_root)
    except (OSError,ValueError,RuntimeError,psycopg.Error) as exc:
        parser.exit(2,f'Backup failed: {exc}\n')


if __name__ == '__main__':
    main()
