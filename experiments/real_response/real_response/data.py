"""Real response data; split/feature fitting never accesses response labels."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

DATA_VERSION = 'real-response-v2-data-1'
OBD_COMMIT = '8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4'


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _download(url: str, path: Path, config: dict, kind: str = '') -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(path.suffix + '.provenance.json')
    if path.exists():
        if meta_path.exists():
            record = json.loads(meta_path.read_text())
            if record['sha256'] != sha256(path):
                raise RuntimeError(f'Cached file hash mismatch: {path}')
            return record
        record = {'requested_url': url, 'resolved_url': None, 'local_import': True,
                  'bytes': path.stat().st_size, 'sha256': sha256(path)}
        _json(meta_path, record)
        return record
    if config.get('offline', False):
        raise FileNotFoundError(f'Offline file missing: {path}')
    limit = int(config.get('runtime', {}).get('max_download_mb', 100) * 1024**2)
    part = path.with_suffix(path.suffix + '.part')
    errors = []
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 90),
                              headers={'User-Agent': 'FisherBalancing-real-response-v2'}) as r:
                r.raise_for_status()
                if int(r.headers.get('content-length', 0)) > limit:
                    raise RuntimeError('Download exceeds max_download_mb')
                size = 0
                with part.open('wb') as f:
                    for block in r.iter_content(1024 * 1024):
                        size += len(block)
                        if size > limit:
                            raise RuntimeError('Download exceeds max_download_mb')
                        f.write(block)
                head = part.read_bytes()[:1024]
                if kind in ('csv', 'parquet', 'zip') and (b'<!DOCTYPE html' in head or b'<html' in head.lower()):
                    raise RuntimeError('HTML received instead of dataset')
                if kind == 'parquet' and head[:4] != b'PAR1':
                    raise RuntimeError('Invalid parquet magic')
                if kind == 'zip' and not zipfile.is_zipfile(part):
                    raise RuntimeError('Invalid ZIP archive')
                os.replace(part, path)
                record = {'requested_url': url, 'resolved_url': r.url,
                          'http_status': r.status_code, 'content_type': r.headers.get('content-type'),
                          'downloaded_utc': datetime.now(timezone.utc).isoformat(),
                          'bytes': size, 'sha256': sha256(path), 'failed_attempts': errors}
                _json(meta_path, record)
                return record
        except Exception as exc:
            errors.append({'attempt': attempt + 1, 'url': url, 'error': repr(exc)})
            if part.exists():
                part.unlink()
            if attempt < 2:
                time.sleep(1 + attempt)
    _json(meta_path, {'requested_url': url, 'failed_attempts': errors, 'status': 'DOWNLOAD_FAILED'})
    raise RuntimeError(f'Download failed: {url}: {errors}')


def _raw_root(config: dict, root: Path) -> Path:
    return Path(config['data_root']) if config.get('data_root') else root / 'data' / 'raw'


def download_all(config: dict, root: Path) -> dict:
    raw = _raw_root(config, root)
    prefix = f'https://raw.githubusercontent.com/st-tech/zr-obp/{OBD_COMMIT}'
    files = {
        'obd/all.csv': (f'{prefix}/obd/random/all/all.csv', 'csv'),
        'obd/item_context.csv': (f'{prefix}/obd/random/all/item_context.csv', 'csv'),
        'obd/README.md': (f'{prefix}/obd/README.md', ''),
        'obd/official_loader.py': (f'{prefix}/obp/dataset/real.py', ''),
        'movielens/ml-100k.zip': ('https://files.grouplens.org/datasets/movielens/ml-100k.zip', 'zip'),
        'lastfm/lastfm-dataset-50.snappy.parquet': ('https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/lastfm-dataset-50.snappy.parquet', 'parquet'),
        'lastfm/userid-profile.tsv.zip': ('https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/userid-profile.tsv.zip', 'zip'),
        'lastfm/README.txt': ('https://github.com/eifuentes/lastfm-dataset-1K/releases/download/v1.0/README.txt', ''),
        'lastfm/release.json': ('https://api.github.com/repos/eifuentes/lastfm-dataset-1K/releases/tags/v1.0', ''),
        'lastfm/original_dataset_information.html': ('http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html', ''),
    }
    from concurrent.futures import ThreadPoolExecutor
    names = set(config.get('datasets', {'obd': {}, 'movielens': {}, 'lastfm': {}}))
    selected = [(name, entry) for name, entry in files.items() if name.split('/')[0] in names]
    def get(item):
        name, (url, kind) = item
        return name, _download(url, raw / name, config, kind)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = dict(executor.map(get, selected))
    _json(root / 'data' / 'download_manifest.json', {'obd_commit': OBD_COMMIT, 'files': records})
    return records


def _zip_read(path: Path, member: str | None = None) -> tuple[bytes, str]:
    with zipfile.ZipFile(path) as z:
        total = 0
        for info in z.infolist():
            p = Path(info.filename)
            if p.is_absolute() or '..' in p.parts or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('Unsafe archive member')
            total += info.file_size
        if total > 256 * 1024**2:
            raise RuntimeError('Archive expanded size exceeds 256 MiB')
        if member is None:
            members = [n for n in z.namelist() if n.endswith('.tsv')]
            if len(members) != 1:
                raise RuntimeError(f'Ambiguous TSV archive: {members}')
            member = members[0]
        return z.read(member), member


def _priority(ids, seed: int, salt: str = '') -> np.ndarray:
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}|{salt}|{i}'.encode()).digest()[:8], 'big')
                     for i in ids], dtype=np.uint64)


def _subsample(indices, ids, cap: int, seed: int, salt: str) -> np.ndarray:
    indices = np.asarray(indices)
    if len(indices) <= cap:
        return indices
    pos = np.argsort(_priority(np.asarray(ids)[indices], seed, salt), kind='stable')[:cap]
    return np.sort(indices[pos])


def _chronological(timestamps) -> tuple[np.ndarray, np.ndarray, dict]:
    t = np.asarray(timestamps)
    order = np.argsort(t, kind='stable')
    boundary = t[order[int(.8 * len(t))]]
    pool = np.flatnonzero(t < boundary)
    test = np.flatnonzero(t >= boundary)
    if not len(pool) or not len(test):
        raise RuntimeError('Chronological split has empty partition')
    return pool, test, {'split_boundary': str(boundary), 'timestamp_ties': 'boundary_group_entirely_test',
                        'pre_subsample_pool_count': len(pool), 'pre_subsample_test_count': len(test)}


def _obd(raw: Path, cfg: dict, seed: int):
    rows = pd.read_csv(raw / 'obd' / 'all.csv', index_col=0)
    items = pd.read_csv(raw / 'obd' / 'item_context.csv', index_col=0)
    if len(rows) != 10000 or not {'timestamp', 'item_id', 'click', 'position'}.issubset(rows.columns):
        raise RuntimeError('Unexpected OBD-small schema/count')
    if rows.index.has_duplicates:
        raise RuntimeError('Duplicate original OBD record IDs')
    if not rows.click.isin([0, 1]).all():
        raise RuntimeError('Invalid original binary click')
    ids = np.array([f'obd:{i}' for i in rows.index])
    frame = rows.merge(items, on='item_id', how='left', validate='many_to_one', sort=False)
    if len(frame) != len(rows):
        raise RuntimeError('OBD item join changed impression count')
    dates = pd.to_datetime(rows.timestamp, utc=True, format='mixed')
    pool, test, meta = _chronological(dates.astype('int64'))
    categorical = [c for c in frame if c.startswith(('user_feature', 'user-item_affinity', 'user_item_affinity'))]
    categorical += ['item_id', 'position', 'item_feature_1', 'item_feature_2', 'item_feature_3']
    numeric = ['item_feature_0']
    audit = {'dataset': 'obd', 'family': 'bernoulli', 'original_rows': len(rows),
             'response': 'original click 0/1', 'original_record_ids_unique': True,
             'affinity_provenance': 'Official pinned README says prior observed user-item clicks',
             'item_feature_0_policy': 'Numeric as treated continuously by pinned official loader',
             'other_anonymized_features_policy': 'categorical; no ordinal interpretation',
             'excluded_predictors': ['click', 'reward', 'timestamp', 'propensity_score', 'action_prob', 'raw_csv_index'],
             'category_counts_pool': {c: int(frame.iloc[pool][c].nunique()) for c in categorical},
             'data_license': 'CC BY 4.0 per OBD paper; official loader code is separately Apache 2.0',
             'source_commit': OBD_COMMIT, **meta}
    aux = {'logging_propensity_pool': rows.propensity_score.to_numpy()[pool],
           'logging_propensity_test': rows.propensity_score.to_numpy()[test]}
    return frame, rows.click.to_numpy(float), ids, pool, test, numeric, categorical, [], audit, aux


def _movielens(raw: Path, cfg: dict, seed: int):
    archive = raw / 'movielens' / 'ml-100k.zip'
    members = ['u.data', 'u.user', 'u.item', 'u.genre', 'u.occupation', 'README']
    files = {m: _zip_read(archive, 'ml-100k/' + m)[0] for m in members}
    rows = pd.read_csv(io.BytesIO(files['u.data']), sep='\t', header=None,
                       names=['user_id', 'item_id', 'rating', 'timestamp'])
    users = pd.read_csv(io.BytesIO(files['u.user']), sep='|', header=None,
                        names=['user_id', 'age', 'gender', 'occupation', 'zip'])
    genres = [line.split('|')[0] for line in files['u.genre'].decode().splitlines() if line]
    genre_cols = ['genre:' + g for g in genres]
    movies = pd.read_csv(io.BytesIO(files['u.item']), sep='|', header=None, encoding='latin-1',
                         names=['item_id', 'title', 'release_date', 'video_release', 'url'] + genre_cols)
    movies['release_year'] = pd.to_datetime(movies.release_date, format='%d-%b-%Y', errors='coerce').dt.year
    frame = rows.merge(users, on='user_id', how='left', validate='many_to_one').merge(
        movies, on='item_id', how='left', validate='many_to_one')
    if len(rows) != 100000 or not rows.rating.between(1, 5).all():
        raise RuntimeError('Unexpected MovieLens 100K ratings/schema')
    ids = np.array([f'ml100k:{i}' for i in rows.index])
    pool, test, meta = _chronological(rows.timestamp)
    pool = _subsample(pool, ids, int(cfg.get('pool_cap', 10000)), seed, 'movielens-pool')
    test = _subsample(test, ids, int(cfg.get('test_cap', 5000)), seed, 'movielens-test')
    frame['gender_male'] = (frame.gender == 'M').astype(float)
    audit = {'dataset': 'movielens', 'family': 'gaussian', 'original_rows': len(rows),
             'response': 'original rating minus fixed public-scale constant 3',
             'raw_rating_scale': [1, 5], 'archive_members': ['ml-100k/' + m for m in members],
             'item_encoding': 'ISO-8859-1 / latin-1',
             'profile_fields': ['age', 'gender', 'occupation'],
             'interaction_rule': 'pool-standardized age x genre, male indicator x genre',
             'excluded_predictors': ['user_id', 'item_id', 'rating', 'timestamp', 'zip', 'title'],
             'data_license': 'GroupLens research-use conditions in archived README; do not redistribute raw data',
             **meta}
    aux = {'original_rating_pool': rows.rating.to_numpy()[pool], 'original_rating_test': rows.rating.to_numpy()[test]}
    (raw / 'movielens' / 'README.txt').write_bytes(files['README'])
    return frame, rows.rating.to_numpy(float) - 3, ids, pool, test, ['age', 'release_year'], ['gender', 'occupation'], genre_cols + ['gender_male'], audit, aux


def _normalize_name(value) -> str:
    if pd.isna(value):
        return ''
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def _lastfm(raw: Path, cfg: dict, seed: int):
    logs = pd.read_parquet(raw / 'lastfm' / 'lastfm-dataset-50.snappy.parquet')
    required = ['user_id', 'timestamp', 'artist_id', 'artist_name']
    if not set(required).issubset(logs.columns):
        raise RuntimeError(f'Unexpected Last.fm schema: {logs.columns.tolist()}')
    profile_bytes, member = _zip_read(raw / 'lastfm' / 'userid-profile.tsv.zip', 'userid-profile.tsv')
    profile = pd.read_csv(io.BytesIO(profile_bytes), sep='\t').rename(columns={'#id': 'user_id'})
    start = pd.Timestamp(cfg.get('eval_start_utc', '2008-05-05T00:00:00Z'))
    history_start = start - pd.Timedelta(days=int(cfg.get('history_days', 180)))
    weeks = int(cfg.get('eval_weeks', 52)); pool_weeks = int(cfg.get('pool_weeks', 42))
    end = start + pd.Timedelta(weeks=weeks)
    logs.timestamp = pd.to_datetime(logs.timestamp, utc=True)
    names = logs.artist_name.map(_normalize_name)
    artist_id = logs.artist_id.fillna('').astype(str).str.strip().str.lower()
    logs['artist_key'] = np.where(artist_id.ne(''), 'mbid:' + artist_id, 'name:' + names)
    missing_artist = artist_id.eq('') & names.eq('')
    history = logs.loc[(logs.timestamp >= history_start) & (logs.timestamp < start)].copy()
    history['history_week'] = ((history.timestamp - history_start).dt.total_seconds() // (7 * 86400)).astype(int)
    elig = history.groupby('user_id').agg(scrobbles=('timestamp', 'size'), active_weeks=('history_week', 'nunique'))
    users = sorted(elig.index[(elig.scrobbles >= cfg.get('history_min_scrobbles', 100)) &
                             (elig.active_weeks >= cfg.get('history_min_active_weeks', 4))].tolist())
    if not users:
        raise RuntimeError('No Last.fm users qualify at fixed history/evaluation dates')
    history = history.loc[history.user_id.isin(users)]
    artist_counts = history.loc[history.artist_key != 'name:'].groupby('artist_key').size()
    ranked = sorted(artist_counts.index, key=lambda a: (-int(artist_counts[a]), a))
    artists = ranked[:int(cfg.get('artist_count', 20))]
    if len(artists) != 20:
        raise RuntimeError(f'Insufficient history artists: {len(artists)}')
    user_counts = history.groupby('user_id').size()
    pair_counts = history.groupby(['user_id', 'artist_key']).size()
    observed = logs.loc[(logs.timestamp >= start) & (logs.timestamp < end) &
                        logs.user_id.isin(users) & logs.artist_key.isin(artists)].copy()
    observed['week'] = ((observed.timestamp - start).dt.total_seconds() // (7 * 86400)).astype(int)
    counts = observed.groupby(['user_id', 'artist_key', 'week']).size()
    grid = pd.MultiIndex.from_product([users, artists, range(weeks)], names=['user_id', 'artist_key', 'week'])
    response = counts.reindex(grid, fill_value=0).to_numpy(dtype=np.int64)
    frame = grid.to_frame(index=False).merge(profile[['user_id', 'age', 'gender', 'country']],
                                           on='user_id', how='left', validate='many_to_one')
    frame['history_user_log1p'] = np.log1p(frame.user_id.map(user_counts).to_numpy(float))
    frame['history_artist_log1p'] = np.log1p(frame.artist_key.map(artist_counts).to_numpy(float))
    pairs = pd.MultiIndex.from_frame(frame[['user_id', 'artist_key']])
    frame['history_pair_log1p'] = np.log1p(pair_counts.reindex(pairs, fill_value=0).to_numpy(float))
    frame['week_sin'] = np.sin(2 * np.pi * frame.week / 52)
    frame['week_cos'] = np.cos(2 * np.pi * frame.week / 52)
    frame['week_trend'] = frame.week.astype(float)
    ids = np.array([f'lastfm:{u}|{a}|{w}' for u, a, w in grid])
    pool = _subsample(np.flatnonzero(frame.week.to_numpy() < pool_weeks), ids, cfg.get('pool_cap', 10000), seed, 'lastfm-pool')
    test = _subsample(np.flatnonzero(frame.week.to_numpy() >= pool_weeks), ids, cfg.get('test_cap', 5000), seed, 'lastfm-test')
    if len(pool) < 480:
        raise RuntimeError(f'Fixed Last.fm subset has inadequate pool size: {len(pool)}')
    audit = {'dataset': 'lastfm', 'display_name': 'Last.fm-1K (50-user subset; weekly counts)',
             'family': 'poisson', 'original_scrobble_rows': len(logs), 'subset_users': int(logs.user_id.nunique()),
             'eligible_user_count': len(users), 'selected_artist_count': len(artists),
             'eligible_users': users, 'selected_artists': artists, 'history_start_utc': str(history_start),
             'eval_start_utc': str(start), 'eval_end_exclusive_utc': str(end), 'pool_weeks': pool_weeks,
             'artist_selection': 'top 20 artists in eligible users history only; normalized-key tie break',
             'history_eligibility': {'min_scrobbles': cfg.get('history_min_scrobbles', 100), 'min_active_weeks': cfg.get('history_min_active_weeks', 4)},
             'history_user_counts': {k: int(v) for k,v in user_counts.items()},
             'history_artist_counts': {k: int(artist_counts[k]) for k in artists},
             'missing_artist_rows': int(missing_artist.sum()),
             'response': 'actual archived integer scrobbles per user/artist/7-day bin; no count transform or clipping',
             'zero_interpretation': 'no archived scrobble in cell; not an observed randomized exposure or dislike',
             'logging_limitation': 'attrition or unrecorded listening may yield archive zeros',
             'profile_archive_member': member, 'exposure_offset': 0,
             'subset_release': 'eifuentes/lastfm-dataset-1K v1.0, processed provenance mirror',
             'original_collector': 'Oscar Celma / MTG-UPF', 'data_terms': 'Preserved original/release README.txt and original information HTML'}
    numeric = ['age', 'history_user_log1p', 'history_artist_log1p', 'history_pair_log1p', 'week_sin', 'week_cos', 'week_trend']
    return frame, response, ids, pool, test, numeric, ['gender', 'country', 'user_id', 'artist_key'], [], audit, {}


def _encode_features(frame: pd.DataFrame, pool: np.ndarray, test: np.ndarray,
                     numeric: list[str], categorical: list[str], binary: list[str], dataset: str):
    """Fit dictionaries, imputation, scaling and interactions using X_pool only."""
    p, t = frame.iloc[pool], frame.iloc[test]
    cols, testcols, artifact = {}, {}, {'numeric': {}, 'categorical': {}, 'rare_threshold': 10, 'fit_scope': 'pool_only'}
    for c in numeric:
        a = pd.to_numeric(p[c], errors='coerce').to_numpy(float)
        b = pd.to_numeric(t[c], errors='coerce').to_numpy(float)
        a[~np.isfinite(a)] = np.nan; b[~np.isfinite(b)] = np.nan
        median = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.
        am = np.isnan(a); bm = np.isnan(b)
        a = np.where(am, median, a); b = np.where(bm, median, b)
        mean = float(a.mean()); std = float(a.std()); scale = std if std > 0 else 1.
        cols['numeric:' + c] = (a - mean) / scale; testcols['numeric:' + c] = (b - mean) / scale
        cols['missing:' + c] = am.astype(float); testcols['missing:' + c] = bm.astype(float)
        artifact['numeric'][c] = {'median': median, 'mean': mean, 'std': scale, 'all_missing': bool(am.all())}
    for c in binary:
        cols['binary:' + c] = p[c].fillna(0).to_numpy(float)
        testcols['binary:' + c] = t[c].fillna(0).to_numpy(float)
    for c in categorical:
        a = p[c].fillna('__MISSING__').astype(str); b = t[c].fillna('__MISSING__').astype(str)
        counts = a.value_counts(); kept = sorted(counts.index[counts >= 10].tolist())
        a = a.where(a.isin(kept), '__OTHER__'); b = b.where(b.isin(kept), '__OTHER__')
        levels = sorted(set(a) | {'__OTHER__'})
        # Drop a pool-observed reference; OTHER remains a usable test code even when absent in pool.
        reference = sorted(set(a))[0]
        artifact['categorical'][c] = {'kept_levels': kept, 'levels': levels, 'reference_level': reference,
                                      'rare_level_count': int((counts < 10).sum()), 'all_levels_rare': not bool(kept),
                                      'test_unseen_or_rare_rows': int((b == '__OTHER__').sum())}
        for level in levels:
            if level == reference:
                continue
            key = f'categorical:{c}={level}'
            cols[key] = (a == level).to_numpy(float); testcols[key] = (b == level).to_numpy(float)
    interactions = []
    if dataset == 'movielens':
        for c in binary:
            if c.startswith('genre:'):
                interactions.extend([('numeric:age', 'binary:' + c), ('binary:gender_male', 'binary:' + c)])
    elif dataset == 'lastfm':
        interactions.extend([('numeric:history_user_log1p', c) for c in cols if c.startswith('categorical:artist_key=')])
    for left, right in interactions:
        key = 'interaction:' + left + '*' + right
        cols[key] = cols[left] * cols[right]; testcols[key] = testcols[left] * testcols[right]
    names = sorted(cols)
    discarded = [n for n in names if np.ptp(cols[n]) <= 1e-14]
    names = [n for n in names if n not in discarded]
    artifact.update(feature_names=names, discarded_constant_columns=discarded, interactions=interactions,
                    dummy_scaling='unscaled 0/1', numeric_scaling='pool mean and population standard deviation')
    A = np.column_stack([cols[n] for n in names]); B = np.column_stack([testcols[n] for n in names])
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise RuntimeError('Nonfinite encoded feature')
    return A, B, artifact


def _pca(A: np.ndarray, B: np.ndarray, components: int = 19):
    # Dense covariance is bounded here (raw dimensions < a few hundred); exact centered eigendecomposition.
    center = A.mean(axis=0)
    Ac = A - center
    covariance = Ac.T @ Ac
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tol = max(A.shape) * np.finfo(float).eps * max(float(values[0]), 1.)
    rank = int(np.sum(values > tol))
    if rank < components:
        raise RuntimeError(f'RANK_DEFICIENT_FEATURES: {rank} raw nonconstant coordinates; {components} required')
    basis = vectors[:, :components].T.copy()
    for i in range(components):
        j = int(np.argmax(np.abs(basis[i])))
        if basis[i, j] < 0:
            basis[i] *= -1
    Z = Ac @ basis.T; Zt = (B - center) @ basis.T
    scale = max(1., float(np.sqrt(1 + np.sum(Z**2, axis=1)).max()))
    X = np.column_stack([np.ones(len(Z)), Z]) / scale
    Xt = np.column_stack([np.ones(len(Zt)), Zt]) / scale
    model = {'center': center, 'components': basis, 'eigenvalues': values[:components], 'scale': np.array(scale)}
    return X, Xt, model, rank


def _target_audit(y) -> dict:
    y = np.asarray(y)
    positive = y[y > 0]
    return {'rows': len(y), 'mean': float(y.mean()), 'variance': float(y.var()),
            'zero_count': int(np.sum(y == 0)), 'zero_fraction': float(np.mean(y == 0)),
            'nonzero_count': int(np.count_nonzero(y)), 'positive_count': int(np.sum(y > 0)),
            'max': float(y.max()), 'min': float(y.min()),
            'positive_quantiles': np.quantile(positive, [0, .25, .5, .75, .9, .99, 1]).tolist() if len(positive) else None}


def _save_npz(path: Path, **arrays) -> None:
    part = path.with_suffix(path.suffix + '.part')
    with part.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(part, path)


def prepare_all(config: dict, root: Path) -> dict:
    root = Path(root)
    downloads = download_all(config, root)
    raw = _raw_root(config, root)
    output = root / 'data' / 'prepared'; output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, cfg in config.get('datasets', {'obd': {}, 'movielens': {}, 'lastfm': {}}).items():
        seed = int(config.get('preprocess_seed', 20261004))
        feature_cfg = {k: v for k, v in cfg.items() if k not in ('additional_labels', 'batch_size', 'candidate_cap')}
        identity = {'version': DATA_VERSION, 'preprocess_seed': seed, 'dataset_config': feature_cfg,
                    'source_hashes': {k: v['sha256'] for k, v in downloads.items() if k.startswith(name + '/')},
                    'data_module_sha256': sha256(Path(__file__)), 'feature_components': 19}
        config_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        meta_path = output / f'{name}.json'; npz_path = output / f'{name}.npz'
        if meta_path.exists() and npz_path.exists():
            old = json.loads(meta_path.read_text())
            if old.get('data_configuration_hash') == config_hash and old.get('artifact_sha256') == sha256(npz_path):
                result[name] = old
                continue
        loader = {'obd': _obd, 'movielens': _movielens, 'lastfm': _lastfm}[name]
        frame, y, ids, pool, test, numeric, categorical, binary, audit, auxiliary = loader(raw, cfg, seed)
        if np.intersect1d(ids[pool], ids[test]).size:
            raise RuntimeError('Pool/test row ID overlap')
        A, B, transformer = _encode_features(frame, pool, test, numeric, categorical, binary, name)
        X, Xt, pca, raw_rank = _pca(A, B)
        d_audit = {}
        for d in [5, 10, 20]:
            singular = np.linalg.svd(X[:, :d], compute_uv=False)
            rank = int(np.linalg.matrix_rank(X[:, :d]))
            warning = []
            if rank < d:
                warning.append('RANK_DEFICIENT_FEATURES')
            if name == 'obd' and (np.sum(y[pool] == 1) < 5*d or np.sum(y[pool] == 0) < 5*d):
                warning.append('LOW_EVENT_REFERENCE')
            d_audit[str(d)] = {'rank': rank, 'singular_values': singular.tolist(), 'warning_codes': warning}
        arrays = {'X_pool': X, 'X_test': Xt, 'y_pool': y[pool], 'y_test': y[test],
                  'row_ids_pool': ids[pool], 'row_ids_test': ids[test], **auxiliary}
        _save_npz(npz_path, **arrays)
        _save_npz(output / f'{name}.transform.npz', **pca)
        _json(output / f'{name}.transform.json', transformer)
        h = hashlib.sha256()
        for a in (X, Xt, ids[pool], ids[test]):
            h.update(np.ascontiguousarray(a).tobytes())
        audit.update(data_version=DATA_VERSION, data_configuration_hash=config_hash,
                     feature_hash=h.hexdigest(), artifact_sha256=sha256(npz_path),
                     data_identity=identity, pool_rows=len(pool), test_rows=len(test), n_pool=len(pool), n_test=len(test),
                     pool_target=_target_audit(y[pool]), test_target=_target_audit(y[test]),
                     feature_mode='centered_pca_unwhitened', feature_components=19,
                     raw_encoded_feature_count=A.shape[1], raw_nonconstant_rank=raw_rank,
                     common_scale=float(pca['scale']), dimensions=d_audit,
                     pool_norm_range=[float(np.linalg.norm(X, axis=1).min()), float(np.linalg.norm(X, axis=1).max())],
                     test_norm_range=[float(np.linalg.norm(Xt, axis=1).min()), float(np.linalg.norm(Xt, axis=1).max())],
                     encoded_test_mean_shift_l2=float(np.linalg.norm(B.mean(0) - A.mean(0))),
                     chronological_shift_note='test transformed with pool imputation, dictionaries, mean, scales and PCA; no test clipping',
                     response_hash=hashlib.sha256(np.ascontiguousarray(np.r_[y[pool], y[test]]).tobytes()).hexdigest())
        _json(meta_path, audit)
        result[name] = audit
    _json(root / 'data' / 'data_audit.json', result)
    return result


def load_dataset(root: Path, name: str, d: int | None = None) -> dict:
    path = Path(root) / 'data' / 'prepared'
    with np.load(path / f'{name}.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if d is not None:
        data['X_pool'] = data['X_pool'][:, :d].copy()
        data['X_test'] = data['X_test'][:, :d].copy()
    data['metadata'] = json.loads((path / f'{name}.json').read_text())
    data['family'] = data['metadata']['family']
    return data
