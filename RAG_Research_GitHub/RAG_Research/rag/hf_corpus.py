"""Pinned HF Parquet provenance, bounded reads, and full-corpus audit.

The HF repack is an explicit corpus variant. It is never labelled as a
byte-verified copy of the original official ZIP.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from .run_cache import file_sha256, make_signature

HF_PROFILE = 'hf_qrecc_parquet'
HF_SCOPE = 'hf_qrecc_parquet_full'
REGISTRY_SHA256 = 'f1a76a4d7a04ec3be3e2d569971c877025d909501d5d76002fbca31573d2ea7d'
AUDIT_VERSION = 2
QUESTION_COUNT = 79952
TEST_COUNT = 16451
# Use exactly Python str.strip's whitespace set, including U+001C..U+001F.
# Arrow's Unicode White_Space set alone differs for those control characters.
PYTHON_WHITESPACE = ''.join(chr(i) for i in range(0x3001) if chr(i).isspace())


def load_release():
    path = Path(__file__).resolve().parents[1] / 'configs/qrecc_hf_passages.json'
    if file_sha256(path) != REGISTRY_SHA256:
        raise ValueError('Trusted HF release registry was modified')
    return json.loads(path.read_text(encoding='utf-8'))


def release_files(release):
    return [{'name': r['path'], 'bytes': r['size'], 'sha256': r['sha256']}
            for r in release['files']]


def validate_files(root, *, hash_files=False):
    root = Path(root).resolve()
    release = load_release()
    expected = {r['path'] for r in release['files']}
    actual = {p.relative_to(root).as_posix() for p in (root / 'data').glob('*.parquet')}
    if actual != expected:
        raise ValueError(f'HF shard set mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}')
    stamps = {}
    for row in release['files']:
        path = root / row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('HF shards must be regular files inside the source directory')
        stat = path.stat()
        if stat.st_size != row['size']:
            raise ValueError('HF shard size mismatch: ' + row['path'])
        if path.with_name(path.name + '.part').exists():
            raise ValueError('Uncommitted HF partial still exists: ' + row['path'])
        if hash_files and file_sha256(path) != row['sha256']:
            raise ValueError('HF shard SHA-256 mismatch: ' + row['path'])
        stamps[row['path']] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return release, stamps


def arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError('HF Parquet requires requirements-full.txt in the experiment environment') from error
    return pa, pq


def parquet_file(path):
    pa, pq = arrow_modules()
    reader = pq.ParquetFile(path, memory_map=False)
    schema = reader.schema_arrow
    for field in ('id', 'contents'):
        if schema.names.count(field) != 1:
            raise ValueError(f'{path}: expected one {field} column')
        if not (pa.types.is_string(schema.field(field).type) or pa.types.is_large_string(schema.field(field).type)):
            raise ValueError(f'{path}: {field} must be a string column')
    return reader


def iter_parquet_rows(path, *, batch_size=1024):
    """Decode at most one batch, retaining original IDs and complete text."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    reader = parquet_file(path)
    for batch in reader.iter_batches(batch_size=batch_size, columns=['id', 'contents'], use_threads=False):
        yield from batch.to_pylist()


def question_labels(path):
    path = Path(path)
    from .config import validate_experiment_data
    manifest = validate_experiment_data({'evaluation_scope': 'full',
        'dataset_manifest_path': str(path.parent / 'manifest.json')}, path)
    splits, required, count = Counter(), set(), 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            count += 1
            splits[row['split']] += 1
            ids = row.get('positive_ids', [])
            if not isinstance(ids, list) or any(not isinstance(pid, str) or not pid.strip() for pid in ids):
                raise ValueError('Malformed question passage annotation')
            required.update(ids)
    if count != QUESTION_COUNT or splits['test'] != TEST_COUNT:
        raise ValueError('HF audit requires every official QReCC question and unchanged test split')
    if dict(splits) != manifest['split_counts']:
        raise ValueError('Question split counts differ from the prepared manifest')
    return {'sha256': file_sha256(path), 'count': count, 'splits': dict(splits)}, required


def audit_hf_corpus(root, questions, *, output=None):
    """Read every row and check all annotated IDs without loading all IDs/text."""
    import pyarrow.compute as pc
    pa, _ = arrow_modules()
    root, questions = Path(root).resolve(), Path(questions).resolve()
    output = Path(output) if output else root / 'hf_corpus_audit.json'
    from build_full_index import atomic_json
    release, stamps = validate_files(root, hash_files=True)
    q_identity, required = question_labels(questions)
    missing = set(required)
    rows_seen = valid = empty = 0
    details = []
    started = time.monotonic()
    for item in release['files']:
        path = root / item['path']
        reader = parquet_file(path)
        file_rows = file_empty = 0
        for batch in reader.iter_batches(batch_size=8192, columns=['id', 'contents'], use_threads=False):
            ids, texts = batch.column(0), batch.column(1)
            if ids.null_count or texts.null_count:
                raise ValueError(f'{item["path"]}: null passage ID/text')
            # Arrow validation is bounded to this batch. Empty text is skipped
            # by the same indexing policy as JSONL; empty IDs are never valid.
            bad_ids = pc.equal(pc.utf8_length(pc.utf8_trim(ids, characters=PYTHON_WHITESPACE)), 0)
            if pc.any(bad_ids).as_py():
                raise ValueError(f'{item["path"]}: blank passage ID')
            empty_text = pc.equal(pc.utf8_length(pc.utf8_trim(texts, characters=PYTHON_WHITESPACE)), 0)
            file_empty += pc.sum(pc.cast(empty_text, pa.int64())).as_py() or 0
            file_rows += batch.num_rows
            if missing:
                # Only the comparatively small union of question annotations is
                # held in memory. Answers/rewrites never enter model inputs here.
                value_set = pa.array(sorted(missing), type=ids.type)
                mask = pc.and_(pc.is_in(ids, value_set=value_set), pc.invert(empty_text))
                missing.difference_update(ids.filter(mask).to_pylist())
        if file_rows != reader.metadata.num_rows:
            raise ValueError('Parquet EOF/metadata row count mismatch: ' + item['path'])
        rows_seen += file_rows
        empty += file_empty
        valid += file_rows - file_empty
        details.append({'name': item['path'], 'rows': file_rows, 'valid_passages': file_rows-file_empty,
                        'empty_text_rows': file_empty})
        print(f'[HF audit] {len(details)}/{len(release["files"])} files; {rows_seen:,} rows; '
              f'{len(missing)} gold IDs remaining', flush=True)
    if validate_files(root)[1] != stamps:
        raise ValueError('HF files changed during the audit')
    if file_sha256(questions) != q_identity['sha256']:
        raise ValueError('Questions changed during the audit')
    proof = {'audit_version': AUDIT_VERSION, 'repo': release['repo'], 'revision': release['revision'],
             'registry_sha256': REGISTRY_SHA256, 'source_fingerprint': make_signature(release_files(release)),
             'questions': q_identity, 'file_count': len(details), 'total_rows': rows_seen,
             'valid_passages': valid, 'empty_text_rows': empty, 'invalid_rows': 0,
             'required_gold_ids': len(required), 'missing_gold_ids': sorted(missing), 'files': details,
             'unique_passage_ids': 'checked by final indexed unique count before the index can be complete',
             'corpus_scope': HF_SCOPE,
             'blank_text_policy': 'Python str.strip() == empty, identical to the indexing iterator',
             'equivalence': 'Pinned publisher repack; not a byte/semantic equivalence claim for the official ZIP'}
    result = {'complete': not missing, 'proof': proof, 'proof_sha256': make_signature(proof),
              'file_stamps': stamps, 'elapsed_seconds': time.monotonic()-started,
              'finished_at': time.strftime('%Y-%m-%d %H:%M:%S %z')}
    atomic_json(output, result)
    if missing:
        raise ValueError(f'{len(missing)} annotated passage IDs are absent; audit saved but corpus is NOT approved')
    from .hf_duplicates import audit_hf_duplicate_ids
    audit_hf_duplicate_ids(root, questions)
    return result


def verified_hf_identity(root, questions, *, hash_files=False, require_duplicate_audit=True):
    """Require completed audit, matching questions, pinned files and provenance."""
    root = Path(root).resolve()
    release, stamps = validate_files(root, hash_files=hash_files)
    path = root / 'hf_corpus_audit.json'
    if not path.is_file():
        raise ValueError('HF corpus has no completed audit; run audit_hf_corpus.py first')
    audit = json.loads(path.read_text(encoding='utf-8'))
    proof = audit.get('proof', {})
    fingerprint = make_signature(release_files(release))
    if (audit.get('complete') is not True or audit.get('proof_sha256') != make_signature(proof)
            or audit.get('file_stamps') != stamps or proof.get('audit_version') != AUDIT_VERSION
            or proof.get('repo') != release['repo'] or proof.get('revision') != release['revision']
            or proof.get('registry_sha256') != REGISTRY_SHA256
            or proof.get('source_fingerprint') != fingerprint or proof.get('corpus_scope') != HF_SCOPE
            or proof.get('file_count') != len(release['files']) or proof.get('invalid_rows') != 0
            or proof.get('missing_gold_ids') != [] or proof.get('valid_passages', 0) < 1):
        raise ValueError('HF audit is incomplete, altered, or stale relative to the current files')
    if (proof.get('total_rows') != proof.get('valid_passages', 0) + proof.get('empty_text_rows', 0)
            or [row.get('name') for row in proof.get('files', [])] != [r['path'] for r in release['files']]
            or any(row.get('valid_passages', -1) < 0 or row.get('empty_text_rows', -1) < 0
                   or row.get('rows') != row['valid_passages'] + row['empty_text_rows']
                   for row in proof.get('files', []))
            or sum(row['rows'] for row in proof.get('files', [])) != proof.get('total_rows')
            or sum(row['valid_passages'] for row in proof.get('files', [])) != proof.get('valid_passages')):
        raise ValueError('HF audit passage counts are inconsistent')
    q_identity, required = question_labels(questions)
    if proof.get('questions') != q_identity or proof.get('required_gold_ids') != len(required):
        raise ValueError('HF audit was produced for different questions/annotations')
    provenance = {key: proof[key] for key in ('repo', 'revision', 'registry_sha256', 'corpus_scope', 'equivalence')}
    provenance.update(audit_sha256=audit['proof_sha256'], expected_documents=proof['valid_passages'])
    identity = {'files': release_files(release), 'source_file_count': len(release['files']),
                'sha256': fingerprint, 'provenance': provenance}
    if require_duplicate_audit:
        from .hf_duplicates import load_duplicate_proof, POLICY
        duplicate, digest = load_duplicate_proof(root, identity)
        provenance.update(raw_valid_passages=proof['valid_passages'],
            expected_documents=duplicate['unique_documents'],
            identical_duplicates=duplicate['identical_duplicates'],
            duplicate_audit_sha256=digest, duplicate_policy=POLICY)
    return identity


def validate_index_corpus(config, identity):
    """Reject corpus variants unless selected explicitly and exactly matched."""
    profile = config.get('full_corpus_source', 'official_zip')
    if profile == 'official_zip':
        if identity.get('corpus_scope') != 'official_qrecc_full':
            raise ValueError('Full evaluation requires a verified official full-corpus index; HF needs explicit configuration')
        from . import official_index
        source = identity.get('source_identity') or {}
        if (not official_index.is_official_archive(source)
                or identity.get('corpus_fingerprint') != source.get('sha256')
                or identity.get('document_key_policy') != official_index.VARIANT_POLICY
                or identity.get('expected_count_argument') != official_index.VALID_ROWS
                or identity.get('expected_documents') != official_index.UNIQUE_VARIANTS):
            raise ValueError('Official index does not match the audited ZIP and distinct ID/text counts')
    elif profile == HF_PROFILE:
        expected = verified_hf_identity(config['corpus_path'], config['eval_path'])
        if (identity.get('corpus_scope') != HF_SCOPE or identity.get('source_identity') != expected
                or identity.get('corpus_fingerprint') != expected['sha256']):
            raise ValueError('Index does not match the selected, audited HF corpus release')
        count = identity.get('expected_documents', identity.get('indexed_passage_count', identity.get('count')))
        if count != expected['provenance']['expected_documents']:
            raise ValueError('HF index count differs from the audited valid-passage count')
    else:
        raise ValueError('Unsupported full_corpus_source')
