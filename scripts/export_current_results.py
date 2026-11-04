"""Export the retained experiment records into a portable publication archive.

Example:
    python scripts/export_current_results.py --source-root /path/to/blind_magpie \
        --manuscript-root /path/to/MAGPIE_blind

The source project is read-only. Raw histories and the twelve published field
archives are copied byte for byte; normalized JSON omits local execution paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

METHODS = ('rpie', 'lsqml', 'gm_rpie', 'gm_magpie')
OVERLAPS = (.75, .875, .9375)
EPOCHS = {.75: 3000, .875: 800, .9375: 200}
FIGURES = ('numerical_synthetic_setup', 'numerical_overlap_convergence',
           'numerical_0875_baseline_objects', 'numerical_noise_summary',
           'numerical_batch_convergence', 'numerical_real_test_pattern',
           'numerical_real_integrated_circuit')
METRICS = ('object_nmse', 'probe_nmse', 'phase_rmse', 'online_amplitude_mse',
           'frozen_amplitude_mse', 'frozen_noiseless_amplitude_mse')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()
                if k not in {'data_reference', 'dataset_reference', 'relative_run_dir'}}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return re.sub(r'(?:/Users/|/storage/|/home/|/scratch/|/private/|/tmp/)[^\s\"\']+',
                      '<local-path>', value)
    return value


def key(result):
    return tuple(result.get(k) for k in ('method', 'overlap_ratio', 'eta', 'batch_size',
                                       'epochs_completed', 'object_nmse'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='Original blind_magpie experiment project (read-only).')
    parser.add_argument('--manuscript-root', type=Path, required=True,
                        help='Final manuscript source folder containing images/.')
    parser.add_argument('--publication-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source, manuscript, publication = (p.resolve() for p in
        (args.source_root, args.manuscript_root, args.publication_root))
    destination = publication / 'results'
    destination.mkdir(parents=True, exist_ok=True)
    noise = read(source / 'results/analysis/noise_short/summary.json')
    batch = read(source / 'results/analysis/batch_0875/summary.json')
    frozen = read(source / 'provenance/short_noise_parameters.json')
    checks = {}
    for mapping in (noise['source_files_sha256'], batch['source_files_sha256']):
        checks.update(mapping)
    for selection in frozen['selections']:
        for candidate in selection['candidates']:
            checks.update(candidate['source_files_sha256'])
    real_hashes = manuscript.parent / 'supporting/real_source_hashes.json'
    if real_hashes.exists():
        checks.update(read(real_hashes))
    for name, expected in checks.items():
        if digest(source / name) != expected:
            raise ValueError(f'Source identity changed: {name}')

    cases, by_source, selected = [], {}, {s: [] for s in ('overlap', 'noise', 'batch', 'real')}
    study_ids = {s: [] for s in selected}
    reconstructions = []

    def add_case(path, study, parameters, is_selected=False):
        path = path.resolve()
        if path in by_source:
            return by_source[path]
        original, config = read(path), read(path.with_name('config.json'))
        if (original['status'] not in ('complete', 'failed') or not original.get('scientific')
                or config.get('smoke') or original['method'] not in METHODS):
            raise ValueError(f'Invalid scientific terminal record: {path}')
        epochs, completed = config['epochs'], original['epochs_completed']
        if original['status'] == 'complete' and completed != epochs:
            raise ValueError(f'Wrong final epoch: {path}')
        if original['status'] == 'failed' and not original.get('error'):
            raise ValueError(f'Missing failure reason: {path}')
        history = path.with_name('history.csv')
        with history.open(newline='') as stream:
            raw = list(csv.DictReader(stream))
        if [int(r['epoch']) for r in raw] != list(range(completed + 1)):
            raise ValueError(f'Incomplete saved history: {path}')
        for metric in ('object_nmse', 'probe_nmse', 'phase_rmse'):
            if original['status'] == 'complete' and metric in original:
                if not math.isfinite(original[metric]) or original[metric] != float(raw[-1][metric]):
                    raise ValueError(f'Saved endpoint differs from history: {path}, {metric}')
        if study == 'real':
            identity = f"{config['dataset']}__{config['method']}"
        else:
            identity = f"{config['method']}__{path.parent.parent.name}__{path.parent.name}"
        case_id = f'{study}__{identity}'
        relative = Path('results/records') / study / identity
        folder = publication / relative
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history, folder / 'history.csv')
        normalized_config = portable(config)
        normalized_config['parameters'] = parameters
        normalized_config['publication_case_id'] = case_id
        write(folder / 'config.json', normalized_config)
        normalized_result = portable(original)
        normalized_result.update(publication_case_id=case_id, parameters=parameters)
        write(folder / 'result.json', normalized_result)
        case = dict(id=case_id, study=study, method=config['method'],
            dataset=config.get('dataset'), overlap_ratio=config.get('overlap_ratio'),
            eta=config.get('eta'), batch_size=config['batch_size'], epochs=epochs,
            epochs_completed=completed, status=original['status'], parameters=parameters,
            selected=is_selected,
            **{metric: original.get(metric) for metric in METRICS},
            files={name: str(relative / name) for name in ('config.json', 'history.csv', 'result.json')},
            source_identity=dict(result_sha256=digest(path), config_sha256=digest(path.with_name('config.json')),
                history_sha256=digest(history), dataset_fingerprint=config.get('dataset_fingerprint'),
                dataset_array_sha256=config.get('dataset_array_sha256'), native_source=config.get('native_source'),
                runner_source_sha256=config.get('runner_source_sha256')))
        # Short aliases keep notebook readers independent of record filenames.
        case['files'] = dict(config=case['files']['config.json'], history=case['files']['history.csv'],
                             result=case['files']['result.json'])
        if original['status'] == 'failed':
            case.update(error=portable(original['error']), failure_epoch=original.get('failure_epoch'))
        cases.append(case)
        by_source[path] = case
        study_ids[study].append(case_id)
        if is_selected:
            selected[study].append(case_id)
        if study == 'real' or (study == 'overlap' and config['overlap_ratio'] == .875 and is_selected):
            reconstructions.append((case, path))
        return case

    for selection in frozen['selections']:
        candidates = selection['candidates']
        method = selection['method']
        expected_grid = ({tuple(sorted(dict(object_alpha=a, probe_alpha=p).items()))
                         for a in (.004,.02,.1) for p in (.004,.02,.1)} if method != 'lsqml' else
                         {tuple(sorted(dict(damping=d,sigma=s).items()))
                         for d in (.04,.1,.2,1) for s in (.3,.4,.5,.6)})
        if {tuple(sorted(c['parameters'].items())) for c in candidates} != expected_grid:
            raise ValueError('The baseline must contain the approved 9/16-pair grids.')
        complete = [c for c in candidates if c['status'] == 'complete']
        best = min(complete, key=lambda c: (c['object_nmse'], c['run_id']))
        if best['run_id'] != selection['source_run_id']:
            raise ValueError('Baseline selection is not the final-object-error minimum.')
        for candidate in candidates:
            add_case(source / candidate['relative_result_path'], 'overlap', candidate['parameters'],
                     candidate['run_id'] == selection['source_run_id'])
    for row in noise['rows']:
        if row['eta'] == .1:
            case = by_source[(source / row['source_result_path']).resolve()]
        else:
            case = add_case(source / row['source_result_path'], 'noise', row['parameters'], True)
        if case['id'] not in study_ids['noise']:
            study_ids['noise'].append(case['id'])
        if case['id'] not in selected['noise']:
            selected['noise'].append(case['id'])
    for row in batch['all_candidates']:
        if row['batch_size'] == 27:
            case = by_source[(source / row['source_result_path']).resolve()]
        else:
            is_selected = row in batch['selected']
            case = add_case(source / row['source_result_path'], 'batch', row['parameters'], is_selected)
        if case['id'] not in study_ids['batch']:
            study_ids['batch'].append(case['id'])
        if row in batch['selected'] and case['id'] not in selected['batch']:
            selected['batch'].append(case['id'])
    for dataset in ('test_pattern', 'integrated_circuit'):
        for method in METHODS:
            path = source / f'evaluations/real/results/runs/{dataset}/{method}/result.json'
            add_case(path, 'real', read(path)['parameters'], True)
    if len(cases) != 247 or Counter(c['study'] for c in cases) != dict(overlap=129,noise=24,batch=86,real=8):
        raise ValueError('Expected 247 unique physical cases.')

    available = list(source.rglob('reconstruction.npz'))
    for case, original_path in reconstructions:
        original = read(original_path)
        matches = []
        for path in available:
            if path.parent.name != original_path.parent.name or not path.with_name('result.json').exists():
                continue
            if key(read(path.with_name('result.json'))) == key(original):
                matches.append(path)
        if not matches:
            raise ValueError(f'Missing published field archive: {case["id"]}')
        path = min(matches, key=lambda p: ('phoenix_incoming' in str(p), len(str(p))))
        relative = Path('results/reconstructions') / f'{case["id"]}.npz'
        target = publication / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if target.stat().st_size >= 100 * 1024 * 1024 or digest(target) != digest(path):
            raise ValueError('A field archive differs from its source or exceeds the file size limit.')
        case['files']['reconstruction'] = str(relative)
        case['source_identity']['reconstruction_sha256'] = digest(target)
        with zipfile.ZipFile(target) as archive:
            case['reconstruction_keys'] = [Path(p.filename).stem for p in archive.infolist()]

    # The small fixture permits CPU figure generation without measured data.
    import numpy as np
    truth_dir = source / 'studies/reference_075/results/datasets/overlap_075_eta_0p1_seed42'
    truth_metadata = read(truth_dir / 'metadata.json')
    truth_names = ('truth', 'probe_truth', 'probe_init', 'roi_mask')
    with np.load(truth_dir / 'dataset.npz', allow_pickle=False) as archive:
        truth_arrays = {name: archive[name] for name in truth_names}
    array_hashes = {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for name, value in truth_arrays.items()}
    if any(array_hashes[name] != truth_metadata['array_sha256'][name] for name in truth_names):
        raise ValueError('The synthetic truth arrays differ from their original data identity.')
    truth_path = destination / 'reconstructions/synthetic_truth.npz'
    np.savez_compressed(truth_path, **truth_arrays)
    with np.load(truth_path, allow_pickle=False) as archive:
        if any(not np.array_equal(archive[name], truth_arrays[name]) for name in truth_names):
            raise ValueError('The compact truth fixture changed an array.')
    synthetic_truth = dict(path=str(truth_path.relative_to(publication)), sha256=digest(truth_path),
        arrays_sha256=array_hashes,
        arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype))
                for name,value in truth_arrays.items()})

    figdir = destination / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    for stem in FIGURES:
        pdf = manuscript / 'images' / f'{stem}.pdf'
        target = figdir / pdf.name
        shutil.copyfile(pdf, target)
        figure_paths.append(str(target.relative_to(publication)))
        png = pdf.with_suffix('.png')
        if png.exists():
            shutil.copyfile(png, figdir / png.name)
        else:
            converter = shutil.which('pdftoppm')
            if converter is None:
                raise RuntimeError('pdftoppm is needed to preview the two real-data PDF figures.')
            subprocess.run([converter, '-f', '1', '-singlefile', '-r', '144', '-png',
                            str(pdf), str(figdir / stem)], check=True)
        figure_paths.append(str((figdir / f'{stem}.png').relative_to(publication)))

    csv_fields = ['id','study','method','dataset','overlap_ratio','eta','batch_size','epochs',
                  'epochs_completed','status','selected','object_alpha','probe_alpha','damping',
                  'sigma',*METRICS,'history','reconstruction']
    lookup = {c['id']: c for c in cases}
    def csv_row(case):
        flat = {k:case.get(k) for k in csv_fields}
        flat.update(case['parameters'])
        flat.update(history=case['files']['history'],reconstruction=case['files'].get('reconstruction'))
        return {k:flat.get(k) for k in csv_fields}
    def write_csv(path, ids):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=csv_fields)
            writer.writeheader();writer.writerows(csv_row(lookup[i]) for i in ids)
    studies = {}
    for study, ids in study_ids.items():
        path = destination / f'{study}_endpoints.csv'
        write_csv(path, ids)
        studies[study] = dict(case_ids=ids,endpoint_csv=str(path.relative_to(publication)),
            count=len(ids),complete=sum(lookup[i]['status']=='complete' for i in ids),
            failed=sum(lookup[i]['status']=='failed' for i in ids))
    selected_ids = list(dict.fromkeys(i for ids in selected.values() for i in ids))
    write_csv(destination/'selected.csv',selected_ids)
    index = dict(schema_version=1, methods=list(METHODS),path_base='repository root',
        endpoint_rule='Prescribed final epoch; lowest final object NMSE among completed candidates. Failed cases are never selected.',
        metric='Aligned complex object NMSE on common ROI [46:978,46:978]; real datasets have no object ground truth.',
        unique_case_count=len(cases), studies=studies, selections=selected,
        selected_csv='results/selected.csv',figures=figure_paths,synthetic_truth=synthetic_truth,cases=cases)
    write(destination/'index.json',index)
    for legacy in ('overlap.csv','noise.csv','real_data.csv'):
        (destination/legacy).unlink(missing_ok=True)
    manifest = {str(p.relative_to(publication)):dict(sha256=digest(p),bytes=p.stat().st_size)
                for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(destination/'manifest.json',dict(schema_version=1,files=manifest,
        unique_case_count=247,reconstruction_archives=len(reconstructions),synthetic_truth_fixture=True,
        source_metadata_files_verified=len(checks)))
    print(json.dumps(dict(cases=len(cases),studies={k:{x:v[x] for x in ('count','complete','failed')} for k,v in studies.items()},selected_counts={k:len(v) for k,v in selected.items()},
                         reconstruction_archives=len(reconstructions),files=len(manifest)),indent=2))


if __name__ == '__main__':
    main()
