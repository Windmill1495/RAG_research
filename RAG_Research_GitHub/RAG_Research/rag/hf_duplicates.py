"""Bounded-memory duplicate audit and lossless HF import policy.

Only a repeated original ID with byte-identical UTF-8 text can be skipped.
Conflicting text or document-key collisions block the entire import.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .run_cache import make_signature

POLICY = 'identical-id-and-text-first-occurrence-v1'


def _key(pid):
    return hashlib.sha256(pid.encode('utf-8')).hexdigest()


def _valid_batches(root):
    from .hf_corpus import load_release, parquet_file, PYTHON_WHITESPACE
    import pyarrow.compute as pc
    for item in load_release()['files']:
        for batch in parquet_file(Path(root)/item['path']).iter_batches(
                batch_size=8192, columns=['id', 'contents'], use_threads=False):
            mask = pc.not_equal(pc.utf8_trim(batch.column(1), characters=PYTHON_WHITESPACE), '')
            yield batch.column(0).filter(mask), batch.column(1).filter(mask)


def _duplicate_keys(path, *, expected_rows, checksum=None):
    previous = None
    duplicates, rows, unique = set(), 0, 0
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            sha.update(line)
            if len(line) != 65 or line[-1:] != b'\n':
                raise ValueError('Malformed sorted document-key file')
            if previous is not None and line < previous:
                raise ValueError('Document-key file is not sorted')
            rows += 1
            if line == previous:
                duplicates.add(line[:64].decode('ascii'))
            else:
                unique += 1
            previous = line
    if rows != expected_rows or (checksum and sha.hexdigest() != checksum):
        raise ValueError('Sorted document-key audit does not match the source')
    return duplicates, unique, sha.hexdigest()


def audit_hf_duplicate_ids(root, questions, *, sorted_keys=None, key_audit=None):
    from .hf_corpus import verified_hf_identity, validate_files
    from build_full_index import atomic_json
    root = Path(root).resolve()
    raw = verified_hf_identity(root, questions, require_duplicate_audit=False)
    expected = raw['provenance']['expected_documents']
    stamps = validate_files(root)[1]
    start = time.monotonic()
    checksum = None
    if sorted_keys is not None:
        if key_audit is None or key_audit.get('source_identity') != raw:
            raise ValueError('Prior key-sort identity differs from the current full audit')
        duplicate_keys, unique, checksum = _duplicate_keys(sorted_keys,
            expected_rows=expected, checksum=key_audit['sorted_keys_sha256'])
    elif expected <= 10000:
        # Fixed small-input budget; never retain all IDs of a large corpus.
        counts = Counter(_key(pid) for ids, _ in _valid_batches(root) for pid in ids.to_pylist())
        if sum(counts.values()) != expected:
            raise ValueError('Raw source count differs from the full audit')
        duplicate_keys, unique = {key for key, count in counts.items() if count > 1}, len(counts)
    else:
        if os.name != 'posix':
            raise RuntimeError('Large HF duplicate audits require GNU sort on the Linux server')
        scratch = root / '.duplicate-audit-work'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            path = Path(temporary) / 'keys.sorted'
            with path.open('wb') as output:
                sorter = subprocess.Popen(['sort', '--parallel=2', '-S', '256M', '-T', temporary],
                    stdin=subprocess.PIPE, stdout=output, env=dict(os.environ, LC_ALL='C'))
                try:
                    for ids, _ in _valid_batches(root):
                        sorter.stdin.write(''.join(_key(pid)+'\n' for pid in ids.to_pylist()).encode('ascii'))
                    sorter.stdin.close()
                    if sorter.wait() != 0:
                        raise RuntimeError('GNU sort failed')
                except BaseException:
                    sorter.terminate()
                    sorter.wait()
                    raise
            duplicate_keys, unique, checksum = _duplicate_keys(path, expected_rows=expected)
    print(f'[HF duplicates] {len(duplicate_keys):,} repeated keys; {unique:,} unique keys', flush=True)
    groups, conflicts, ordinal = {}, [], 0
    for ids, texts in _valid_batches(root):
        for offset, pid in enumerate(ids.to_pylist()):
            ordinal += 1
            key = _key(pid)
            if key not in duplicate_keys:
                continue
            text_sha = hashlib.sha256(texts[offset].as_py().encode('utf-8')).hexdigest()
            if key not in groups:
                groups[key] = {'id': pid, 'text_sha256': text_sha, 'first_valid_ordinal': ordinal,
                               'skip_valid_ordinals': []}
            else:
                first = groups[key]
                if first['id'] != pid or first['text_sha256'] != text_sha:
                    conflicts.append({'document_key': key, 'valid_ordinal': ordinal,
                                      'id': pid, 'text_sha256': text_sha})
                else:
                    first['skip_valid_ordinals'].append(ordinal)
        if ordinal % 1000000 < 8192:
            print(f'[HF duplicates] checked {ordinal:,} valid rows', flush=True)
    skipped = sum(len(group['skip_valid_ordinals']) for group in groups.values())
    passed = not conflicts and ordinal == expected and unique+skipped == expected
    proof = {'version': 1, 'policy': POLICY, 'source_fingerprint': raw['sha256'],
        'raw_audit_sha256': raw['provenance']['audit_sha256'], 'raw_valid_passages': expected,
        'unique_documents': unique, 'identical_duplicates': skipped,
        'conflict_count': len(conflicts), 'conflicts': conflicts,
        'groups': dict(sorted(groups.items())), 'sorted_document_keys_sha256': checksum}
    if validate_files(root)[1] != stamps:
        raise ValueError('HF files changed during duplicate audit')
    result = {'complete': passed, 'proof': proof, 'proof_sha256': make_signature(proof),
              'elapsed_seconds': time.monotonic()-start, 'finished_at': time.strftime('%Y-%m-%d %H:%M:%S %z')}
    atomic_json(root/'hf_duplicate_audit.json', result)
    if not passed:
        raise ValueError('HF duplicate ID/text conflict or inconsistent counts; import is blocked')
    return result


def load_duplicate_proof(root, raw_identity):
    path = Path(root)/'hf_duplicate_audit.json'
    if not path.is_file():
        raise ValueError('HF needs a completed duplicate audit; run audit_hf_corpus.py')
    value = json.loads(path.read_text(encoding='utf-8'))
    proof = value.get('proof', {})
    groups = proof.get('groups', {})
    raw = raw_identity['provenance']
    if (value.get('complete') is not True or value.get('proof_sha256') != make_signature(proof)
            or proof.get('version') != 1 or proof.get('policy') != POLICY
            or proof.get('source_fingerprint') != raw_identity['sha256']
            or proof.get('raw_audit_sha256') != raw['audit_sha256']
            or proof.get('raw_valid_passages') != raw['expected_documents']
            or proof.get('conflict_count') != 0 or proof.get('conflicts') != []
            or not isinstance(groups, dict)):
        raise ValueError('HF duplicate audit is incomplete, conflicting, altered, or stale')
    skips = []
    for key, group in groups.items():
        first, offsets = group.get('first_valid_ordinal'), group.get('skip_valid_ordinals')
        digest = group.get('text_sha256')
        if (not isinstance(group.get('id'), str) or _key(group['id']) != key
                or not isinstance(digest, str) or len(digest) != 64
                or type(first) is not int or first < 1 or not isinstance(offsets, list) or not offsets
                or any(type(n) is not int or not first < n <= raw['expected_documents'] for n in offsets)):
            raise ValueError('Malformed duplicate positions')
        skips.extend(offsets)
    if (len(set(skips)) != len(skips) or len(skips) != proof.get('identical_duplicates')
            or type(proof.get('unique_documents')) is not int or proof['unique_documents'] < 1
            or proof['unique_documents']+len(skips) != raw['expected_documents']):
        raise ValueError('Duplicate audit counts/positions are inconsistent')
    return proof, value['proof_sha256']


def iter_index_passages(root, questions):
    from .hf_corpus import verified_hf_identity, validate_files
    from .full_data import iter_full_passages
    raw = verified_hf_identity(root, questions, require_duplicate_audit=False)
    proof, _ = load_duplicate_proof(root, raw)
    stamps = validate_files(root)[1]
    skips = {n for group in proof['groups'].values() for n in group['skip_valid_ordinals']}
    expected_rows = {n: group for group in proof['groups'].values()
                     for n in [group['first_valid_ordinal'], *group['skip_valid_ordinals']]}
    ordinal = yielded = 0
    for ordinal, row in enumerate(iter_full_passages(Path(root)), 1):
        if ordinal in expected_rows:
            group = expected_rows[ordinal]
            if (row['id'] != group['id']
                    or hashlib.sha256(row['text'].encode('utf-8')).hexdigest() != group['text_sha256']):
                raise ValueError('Duplicate position no longer contains the audited ID/text')
        if ordinal in skips:
            continue
        yielded += 1
        yield row
    if (ordinal != proof['raw_valid_passages'] or yielded != proof['unique_documents']
            or validate_files(root)[1] != stamps):
        raise ValueError('HF source count/state changed during deduplicated import')
