"""Record model and interpreter identity for each new job's report."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import os
import json

ROOT=Path(__file__).resolve().parents[1]


def execution_manifest(state):
    folders = [ROOT/'models/deployable_feature_profiles'/state['profile']] if state['kind']=='raw' else [
        ROOT/'models/joint_arabidopsis_rice_common6751',
        ROOT/'models'/('arabidopsis_single_strict2601_common6751' if state['species']=='arabidopsis' else 'rice_single_strict399_Tos17N4_common6751')]
    rows=[]
    for folder in folders:
        for path in sorted(p for p in folder.iterdir() if p.suffix in {'.joblib','.json','.tsv'}):
            with path.open('rb') as handle:
                digest=hashlib.file_digest(handle,'sha256').hexdigest()
            rows.append({'artifact':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':digest})
    versions={}
    for package in ['numpy','pandas','scikit-learn','lightgbm','xgboost','torch','transformers']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='not installed'
    references=[]
    resource=Path(os.environ.get('PLANT_EG_REFERENCE',str(ROOT/'webapp_data/nar_reference')))
    paths=[resource/f"{state['species']}{suffix}" for suffix in ['.dmnd','.fasta','.labels.tsv']] if state['kind']=='raw' else [ROOT/'predictions/publication_release'/('Arabidopsis_master_prediction_table.tsv.gz' if state['species']=='arabidopsis' else 'Rice_master_prediction_table.tsv.gz')]
    for path in paths:
        if path.exists():
            with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
            references.append({'artifact':path.name,'sha256':digest})
    code=[]
    for path in sorted((ROOT/'webapp').glob('nar_*.py')):
        code.append({'artifact':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for name in ['scripts/prediction/predict_from_processed_features.py',
                 'scripts/feature_extraction/raw_upload_to_profile_features.py']:
        path=ROOT/name
        if path.exists():
            code.append({'artifact':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'python':platform.python_version(),'package_versions':versions,'model_artifacts':rows,'reference_artifacts':references,
            'code_artifacts':code,
            'manifest_scope':'Hashes identify current on-disk files at job execution, not a historical freeze. Protein-language-model weights and all annotation snapshots are not fully covered.',
            'phenotype_scope':'Original study severe loss-of-function definitions; not a strict lethality classifier'}
